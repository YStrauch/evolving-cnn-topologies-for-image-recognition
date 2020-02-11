'''This module creates a diagram dataset yaml that can be used to visualise a GA'''
import math
from collections import OrderedDict

from matplotlib import cm
import yaml
import numpy as np

from cga_src.ga import GA
from cga_src.cnn import CNN
from cga_src.evaluator import Evaluator
from cga_src.pooling_layer import PoolingLayer
from cga_src.population import Population


class DiagramExporter:
    '''Exports a pickle from a GA to a yaml file that can be used to visualise the GA history'''

    def __init__(self):
        self._point_id = 0
        self._colors = None
        self._species = None
        self._generations = None
        self._algorithm = None

    def pickle_to_yaml(self, data, file_in, file_out):
        '''Parses a pickle to a diagram yaml file'''
        algorithm = GA.from_pickle(
            file_in,
            data=data,
            restore_individual=lambda top: CNN.from_string(
                top,
                input_shape=data.example.shape,
                num_classes=data.num_classes
            ),
        )

        # unset pickle file so that it won't start pickling again
        algorithm.pickle_file = None

        return self.ga_to_yaml(algorithm, file_out)

    def ga_to_yaml(self, algorithm, file_out):
        '''Parses a GA history to a diagram yaml file'''
        self._point_id = 0
        self._algorithm = algorithm

        self._generations = [gen for gen in algorithm.generations]

        # This updates old pickles where pooling layers had a filter depth
        evaluator = algorithm.evaluator

        def update_individual(individual: CNN):
            # pylint: disable=protected-access
            topology_before = individual.topology
            for layer in individual._layers:
                if isinstance(layer, PoolingLayer):
                    layer._kernel._depth = None

            if topology_before in evaluator.cache:
                evaluator.cache[individual.topology] = evaluator.cache[topology_before]

            if individual.history.parent1:
                update_individual(individual.history.parent1)
            if individual.history.parent2:
                update_individual(individual.history.parent2)

        for gen in self._generations:
            for individual in gen:
                update_individual(individual)
        self._assign_species()

        points = {}
        for gen_index, gen in enumerate(self._generations):
            for i in gen.individuals:
                self._parse_individual(i, gen_index+1, None, points)

        # get the best individual
        best_individual = algorithm.best_individual

        # and the one with the highest fitness
        highest_acc_index = np.argmax([
            ev.accuracy for gen in algorithm.generations for ev in gen.evaluations
        ])

        highest_acc_individual = [
            ind for gen in algorithm.generations for ind in gen.individuals
        ][highest_acc_index]

        # parse the two
        best_individual_gen_index = highest_acc_gen_index = None
        for i, gen in enumerate(algorithm.generations):
            if best_individual_gen_index is None and best_individual in gen:
                best_individual_gen_index = i + 1
            if highest_acc_individual is not None and highest_acc_gen_index is None and highest_acc_individual in gen:
                highest_acc_gen_index = i + 1
            if best_individual_gen_index is not None and highest_acc_gen_index is not None:
                break

        best_individual = self._parse_individual(
            best_individual,
            best_individual_gen_index,
            None,
            points
        )
        highest_acc_individual = self._parse_individual(
            highest_acc_individual,
            highest_acc_gen_index,
            None,
            points
        )

        # unpack the structure into a flat array
        points = [p for point in points.values() for data in point.values()
                  for p in data]
        # sort by z index so frontend can just iterate over the points
        points = sorted(points, key=lambda i: i['z'], reverse=True)

        min_fitness = min([gen.worst_fitness
                           for gen in algorithm.generations]
                          )
        obj = {
            'points': points,
            'num_generations': len(self._generations),
            'max_fitness': algorithm.best_fitness,
            'min_fitness': min_fitness,
            'best_individual': best_individual,
            'highest_acc_individual': highest_acc_individual,
            'evaluation_time': algorithm.evaluation_time
        }

        with open(file_out, 'w') as file:
            yaml.safe_dump(obj, file)

    def _assign_species(self):
        rgb = cm.get_cmap('tab10')
        greys = cm.get_cmap('binary')

        all_topologies = [i.topology for g in self._generations for i in g]
        self._species = {topology: i for
                         (i, topology) in enumerate(all_topologies)}

        all_topologies = set(all_topologies)
        fitnesses = {}
        accuracies = {}

        # if individuals have been evaluated with multiple epochs: take the highest fitness
        for topology in {i.topology for g in self._generations for i in g}:
            fitnesses[topology] = max([
                fitnesses[topology] if topology in fitnesses else -1,
                self._algorithm.evaluator.get_evaluation(
                    topology
                ).result().fitness
            ])
            accuracies[topology] = max([
                accuracies[topology] if topology in accuracies else -1,
                self._algorithm.evaluator.get_evaluation(
                    topology
                ).result().accuracy
            ])

        # get the top N topologies according to accuracy
        top_n = rgb.N
        accuracies = sorted(OrderedDict(accuracies),
                            key=lambda i: accuracies[i],
                            reverse=True
                            )
        top_individuals = accuracies[:top_n]

        def hex_color(float_color):
            float_color = float_color[:3]
            rgb = tuple([int(256 * v) for v in float_color])
            return '#%02x%02x%02x' % rgb

        # create colors for top topologies
        # using ordered dict so the assignment of colors will be the same for every pickle
        self._colors = OrderedDict({
            t: hex_color(rgb.colors[i]) for i, t in enumerate(top_individuals)
        })

        # for all others use greyscale
        number_of_greys = len(accuracies) - \
            len(top_individuals)

        self._colors.update(OrderedDict({
            t: hex_color(greys(i / number_of_greys)[:3])
            for i, t in enumerate(all_topologies.difference(top_individuals))
        }))

    def _parse_individual(self, individual: CNN, gen_index, child: CNN, points):
        '''
        Recursively iterates through the history and extracts
        X,Y,Z coordinates and colors from it
        '''

        fitness = individual.history.evaluation.fitness if individual.history.evaluation is not None \
            else child.history.evaluation.fitness if child and child.history.evaluation else -1

        if gen_index not in points:
            points[gen_index] = {}
        if fitness not in points[gen_index]:
            points[gen_index][fitness] = []
        else:
            for point in points[gen_index][fitness]:
                # prevent multi evaluation of the same data point
                if individual.topology == point['topology']:
                    return point

        points_at_this_location = len(
            points[gen_index][fitness]
        )

        def get_parent_generation(own_generation, parent: CNN):
            if parent.history.evaluation is None:
                # this parent was never in a generation (crossover-mutation by product)
                # make him a ghost
                return own_generation - .5
            # else: this is a proper parent generation, so return a whole number that is smaller
            if own_generation % 1 == 0:
                return own_generation - 1
            return math.floor(own_generation)

        parent1 = parent2 = None
        if individual.history.parent1 is not None:
            parent_generation = get_parent_generation(
                gen_index,
                individual.history.parent1
            )
            parent1 = self._parse_individual(
                individual.history.parent1,
                parent_generation,
                individual, points
            )

        if individual.history.parent2 is not None:
            parent_generation = get_parent_generation(
                gen_index,
                individual.history.parent2
            )
            parent2 = self._parse_individual(
                individual.history.parent2,
                parent_generation,
                individual, points
            )

        point = dict(individual.history.__dict__)
        color = self._colors[individual.topology] if individual.topology in self._colors else None

        if individual.topology not in self._species:
            self._species[individual.topology] = 'ghost_%d' % len(
                self._species)

        point.update({
            'id': 'p_%d' % self._point_id,
            'species': 's_%s' % str(self._species[individual.topology]),

            'x': gen_index,
            'y': fitness,
            'z': points_at_this_location * 4,
            'evaluation': individual.history.evaluation.__dict__ if individual.history.evaluation else None,
            'topology': individual.topology,
            # epoch could be a np int which confuses the yaml parser
            'num_epoch': int(individual.num_epochs),
            'color': color,
            'parent1': parent1,
            'parent2': parent2,
            'num_params': sum(p.numel() for p in individual.parameters()),
        })
        self._point_id += 1
        points[gen_index][fitness] += [point]

        return point
