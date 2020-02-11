import warnings
import os
import threading
import pickle
import numpy as np

from cga_src.base import AbstractSearchSpace, CannotReduceResolutionError
from cga_src.population import Population
from cga_src.evaluator import Evaluator


class GA():
    def __init__(self,
                 evaluator,
                 create_individual,
                 epoch_fn=None,
                 population_size=20,
                 generations=None,
                 crossover_probability=.9,
                 mutation_probability=.2,
                 pickle_file=None
                 ):

        assert isinstance(evaluator, Evaluator)

        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.pickle_lock = threading.Lock()

        self.epoch_fn = epoch_fn or self._epoch_fn_not_implemented

        self._create_individual = create_individual
        self.evaluator = evaluator

        # makes the evaluator pickle the GA everytime a new result is there
        if pickle_file:
            self.evaluator.progress_cb = lambda: self.pickle(
                pickle_file, wait=False)

        self._population_size = population_size
        if generations:
            assert isinstance(generations, list)

            for gen in generations:
                assert isinstance(gen, Population)

            self.generations = generations
        else:
            individuals = [
                create_individual() for individual in range(population_size)
            ]

            self.generations = [Population(evaluator, individuals=individuals)]

        self._assign_train_epochs(
            self.latest_generation, len(self.generations) - 1)

    def _epoch_fn_not_implemented(self, individual, gen_index, prev_evaluations):
        raise NotImplementedError(
            'Epoch function has not been set in the GA'
        )

    def _assign_train_epochs(self, generation, gen_index):
        # pylint: disable=comparison-with-callable
        if not self.epoch_fn or self.epoch_fn == self._epoch_fn_not_implemented:
            return

        prev_gen = self.generations[gen_index - 1] if gen_index else None
        if prev_gen:
            prev_gen.recalc()  # ensure that the evaluations are actually there

        for ind in generation.individuals:
            ind.num_epochs = self.epoch_fn(
                gen_index=gen_index,
                prev_evaluations=prev_gen.evaluations if prev_gen else None,
                individual=ind
            )

    @property
    def latest_generation(self):
        gen = self.generations[-1]
        assert isinstance(gen, Population)
        return gen

    @property
    def best_individual(self):
        # if elitism is false, the best individual might be in any generation
        best_fitnesses = [gen.best_fitness for gen in self.generations]
        return self.generations[np.argmax(best_fitnesses)].best_individual

    @property
    def evaluation_time(self):
        return sum([gen.evaluation_time for gen in self.generations])

    @property
    def best_fitness(self):
        # if elitism is false, the best individual might be in any generation
        return max([gen.best_fitness for gen in self.generations])

    def pickle(self, fname=None, wait=True):
        '''
        Returns a pickleable representation and optionally saves to file
        wait does two things: wait for promises to resolve and wait for the lock
        '''

        if fname:
            # pylint: disable=assignment-from-no-return
            acquired = self.pickle_lock.acquire(wait)
            if not acquired:
                return None

        for gen in self.generations:
            for ind, evaluation in zip(gen.individuals, gen.evaluations):
                if ind.history:
                    ind.history.evaluation = evaluation

        generations = [
            [(individual.topology, individual.history)
             for individual in gen.individuals]
            for gen in self.generations
        ]

        dump = {
            'evaluator': self.evaluator.pickle(wait=wait),
            'population_size': self._population_size,
            'generations': generations,
            'crossover_probability': self.crossover_probability,
            'mutation_probability': self.mutation_probability,
        }

        if fname:
            # writes to a tmp file first to prevent destroying the old pickle
            # in case something goes wrong (i.e. we lose write rights on yann)
            pickle.dump(dump, open(fname + '_tmp', 'wb'))
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

            os.rename(fname+'_tmp', fname)

            self.pickle_lock.release()

        return dump

    @classmethod
    def from_pickle(cls, pckle, data, restore_individual, epoch_fn=None):
        '''
        Restores a pickleable representation from either a file or an object
        '''

        pickle_file = None

        if isinstance(pckle, str):
            dump = pickle.load(open(pckle, 'rb'))
            pickle_file = pckle
        else:
            dump = pckle

        evaluator = Evaluator.from_pickle(dump['evaluator'], data=data)

        generations = []
        for gen in dump['generations']:
            individuals = []
            for topology, history in gen:
                individual = restore_individual(topology)
                individual.history = history
                individuals.append(individual)

            gen = Population(evaluator=evaluator,
                             individuals=individuals)

            generations.append(gen)

        return cls(
            evaluator=evaluator,
            population_size=dump['population_size'],
            generations=generations,
            crossover_probability=dump['crossover_probability'],
            mutation_probability=dump['mutation_probability'],
            epoch_fn=epoch_fn,
            create_individual=None,
            pickle_file=pickle_file
        )

    def _truncate_to_popsize(self, population):
        assert isinstance(population, Population)

        while len(population) > self._population_size:
            del population[population.index(population.worst_individual)]

        return population

    def evolve(self, search_space, elitism=True):
        old_gen = self.generations[-1]
        old_gen.recalc()

        # the population might be bigger than pop size because of elitism and
        # lyceum stopping us prematurely during elitism logic
        # we will therefore enforce population size now
        old_gen = self._truncate_to_popsize(old_gen)

        new_gen = Population(self.evaluator)

        assert isinstance(search_space, AbstractSearchSpace)
        assert isinstance(old_gen, Population)

        while len(new_gen) < self._population_size:
            individual1 = old_gen.tournament_selection()
            individual2 = old_gen.tournament_selection()
            if np.random.rand() <= self.crossover_probability:
                remaining_tries = 5000
                while remaining_tries > 0:
                    remaining_tries -= 1
                    try:
                        individual1, individual2 = individual1.one_point_crossover(
                            individual2
                        )
                        remaining_tries = 0
                    except CannotReduceResolutionError:
                        if remaining_tries == 0:
                            warnings.warn(
                                "Warning: Cannot reduce resolution during cross-over. " +
                                "Skipping cross-over."
                            )

            if np.random.rand() <= self.mutation_probability:
                individual1 = individual1.mutate(search_space)
                individual2 = individual2.mutate(search_space)

            # make sure that new individual references are added
            if individual1 in old_gen:
                individual1 = individual1.clone_with_weights()
            if individual2 in old_gen:
                individual2 = individual2.clone_with_weights()

            new_gen += [individual1, individual2]

        if elitism and old_gen.best_individual not in new_gen:
            new_gen += [old_gen.best_individual.clone_with_weights()]

        self._assign_train_epochs(new_gen, len(self.generations))

        new_gen = self._truncate_to_popsize(new_gen)
        self.generations += [new_gen]
