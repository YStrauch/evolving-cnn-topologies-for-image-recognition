'''Fitness evaluation for CNN topology strings'''
import concurrent.futures
import logging
import pickle
import time

import torch
import torch.nn as nn

from cga_src.hardware import HardwareManager
from cga_src.cnn import CNN
from cga_src.data import TorchDataset
from cga_src.base import InvalidTopologyError


class Evaluation:
    def __init__(self, accuracy, duration, size, fitness_punishment_per_hour=0):
        self.size = size
        self.accuracy = accuracy
        self.fitness = accuracy - duration * fitness_punishment_per_hour
        self.duration = duration


class Evaluator():
    '''
    Each evaluator is bound to a data set.
    It caches and evaluates fitnesses of CNN topology strings.
    '''

    def __init__(self,
                 data,
                 cache=None,

                 fitness_punishment_per_hour=0,
                 learning_rate_decay_points=None,
                 learning_rate=0.1,
                 momentum=0.9,
                 learning_rate_decay_factor=0.9,
                 ):
        self.cache = cache or {}
        self._data = data
        self._learning_rate_decay_points = learning_rate_decay_points or []
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._learning_rate_decay_factor = learning_rate_decay_factor
        self.progress_cb = lambda: None
        self.fitness_punishment_per_hour = fitness_punishment_per_hour

        assert isinstance(self.cache, dict)
        assert isinstance(self._data, TorchDataset)

    def pickle(self, fname=None, wait=True):
        '''
        Returns a pickleable representation and optionally saves to file
        By default it will wait for all futures to complete
        '''

        pickleable_cache = {}
        # thread safe: don't iterate over self.cache directly
        for topology in list(self.cache.keys()):
            evaluation = self.cache[topology]['evaluation']

            if isinstance(evaluation, concurrent.futures.Future):
                if (evaluation.done() or wait):
                    evaluation = evaluation.result()
                else:
                    continue
            model = self.cache[topology]['model'] \
                .to('cpu') if self.cache[topology]['model'] else None

            pickleable_cache[topology] = {
                'evaluation': evaluation,
                'model': model
            }

        obj = {
            'cache': pickleable_cache,
            'learning_rate_decay_points': self._learning_rate_decay_points,
            'learning_rate': self._learning_rate,
            'momentum': self._momentum,
            'fitness_punishment_per_hour': self.fitness_punishment_per_hour,
            'learning_rate_decay_factor': self._learning_rate_decay_factor,
        }

        if fname:
            with open(fname, 'wb') as file:
                pickle.dump(obj, file)

        return obj

    @classmethod
    def from_pickle(cls, pckle, data):
        '''
        Restores a pickleable representation from either a file or an object
        '''

        if isinstance(pckle, str):
            with open(pckle, 'rb') as file:
                pckle = pickle.load(file)

        parsed_cache = {}
        cache = pckle['cache']
        for topology in cache:
            future = concurrent.futures.Future()

            # backwards compatability: old pickles had no 'evaluation'
            if 'evaluation' in cache[topology]:
                evaluation = cache[topology]['evaluation']
            else:
                evaluation = cache[topology]['fitness'], cache[topology]['fitness']

            # yet another backwards compatibility: create the object if it is a tuple
            if not isinstance(evaluation, Evaluation):
                fitness, accuracy = evaluation
                evaluation = Evaluation(accuracy, 0, 0)
                evaluation.fitness = fitness

            future.set_result(evaluation)
            parsed_cache[topology] = {
                'evaluation': future,
                'model': cache[topology]['model']
            }

        fitness_punishment_per_hour = 0 if 'fitness_punishment_per_hour' not in pckle else pckle[
            'fitness_punishment_per_hour']

        return cls(cache=parsed_cache,
                   data=data,
                   learning_rate_decay_points=pckle['learning_rate_decay_points'],
                   learning_rate=pckle['learning_rate'],
                   momentum=pckle['momentum'],
                   learning_rate_decay_factor=pckle['learning_rate_decay_factor'],
                   fitness_punishment_per_hour=fitness_punishment_per_hour,
                   )

    def get_evaluation(self, individual):
        '''
        Gets a tuple of (fitness, accuracy) of a topology as a promise.
        Call .result() to block for it
        '''

        if isinstance(individual, str):
            # this makes sure that the cache is not invalidated by small differences
            # when making manual changes on string topologies
            model = CNN.from_string(
                input_shape=self._data.example.shape,
                string=individual,
                num_classes=self._data.num_classes
            )
        else:
            model = individual

        if model.topology in self.cache:
            # this topology was previously trained for this number of epochs
            return self.cache[model.topology]['evaluation']

        evaluation = None
        # try to find a model that has been partially trained
        required_epoch = model.num_epochs
        for available_epoch in range(required_epoch - 1, -1, -1):
            partially_trained_individual = model.clone()
            partially_trained_individual.num_epochs = available_epoch
            if partially_trained_individual.topology in self.cache:
                # this topology was previously trained for a smaller number of epochs
                entry = self.cache[partially_trained_individual.topology]

                # edge case: find a model that finished training
                if not entry['evaluation'].done():
                    continue

                evaluation = entry['evaluation'].result()
                assert isinstance(evaluation, Evaluation)

                # resume training: query and clone the partially trained model with weights
                model = entry['model'].clone_with_weights()
                model.num_epochs = required_epoch

                evaluation = HardwareManager.add_job(self._train_test, {
                    'model': model,
                    'start_epoch': available_epoch,
                    'end_epoch': model.num_epochs,
                    'duration': evaluation.duration,
                })
                break

        if not evaluation:
            # this model has not been partially trained, so train from the beginning
            evaluation = HardwareManager.add_job(self._train_test, {
                'model': model,
                'start_epoch': 0,
                'end_epoch': model.num_epochs,
                'duration': 0,
            })

        self.cache[model.topology] = {
            'model': model,
            'evaluation': evaluation
        }

        def callback(_):
            time.sleep(1)
            self.progress_cb()

        evaluation.add_done_callback(callback)

        return evaluation

    def get_evaluations(self, population):
        '''Generator that yields pairs of (topology, evaluation)
        as they are evaluated by the hardware'''

        promises = [self.get_evaluation(individual)
                    for individual in population]
        future_mapping = {promises[i]: population[i]
                          for i in range(len(population))}

        for promise in concurrent.futures.as_completed(promises):
            topology = future_mapping[promise]
            evaluation = promise.result()

            yield topology, evaluation

    def _train_test(self, model, start_epoch, end_epoch, duration, device):

        try:
            model, duration_train = self._train(
                model, start_epoch, end_epoch, device
            )
            accuracy, duration_test = self._test(model, device)
            duration += (duration_train + duration_test) / 3600  # in hours

            model = model.to('cpu')  # try to free some space

            return Evaluation(accuracy, duration, model.size, self.fitness_punishment_per_hour)

        except RuntimeError as exc:
            if 'out of memory' in str(exc):
                logging.warning('Ran out of memory for GPU (%s), individual %s',
                                device, model.topology)
                return Evaluation(0, 0, model.size)

            if 'Given groups' in str(exc):
                raise InvalidTopologyError(
                    model.topology,
                    model._layers[-1].output_shape,  # pylint: disable=protected-access
                    exc.args[0]
                )

            raise exc

    def _train(self,
               model,
               start_epoch,
               end_epoch,
               device
               ):

        model = model.to(device)

        start_time = time.time()
        model.train()

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self._learning_rate,
            momentum=self._momentum
        )

        logging.debug('Start train on %s, start epoch %d, end epoch %d',
                      device, start_epoch, end_epoch)
        # Train the model

        learning_rate = self._learning_rate
        for epoch in range(start_epoch, end_epoch):
            if epoch in self._learning_rate_decay_points:
                learning_rate *= self._learning_rate_decay_factor
                # not sure if I am allowed to do this
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate, momentum=self._momentum)

            for i, (images, labels) in enumerate(self._data.train):
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 300 == 0:
                    total_step = len(self._data.train)
                    logging.debug('Epoch [%d/%d], Step [%d/%d], Loss: %.4f}',
                                  epoch+1, end_epoch,
                                  i+1, total_step, loss.item())
        # Save the model checkpoint
        # torch.save(model.state_dict(), 'model.ckpt')

        return model, time.time() - start_time

    def _test(self, model, device):
        model = model.to(device)

        # Test the model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            start_time = time.time()
            for images, labels in self._data.test:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.debug('Ended train on %s', device)
        return correct / total, time.time() - start_time
