# pylint: disable=protected-access, missing-docstring

import unittest
import helpers
import cga


class GA(helpers.Test):
    def setUp(self):
        self.data = cga.MNIST()

        self.topology_index = 0
        self.topologies = [
            'C[001(2x2 2x2 V)]\nC[001(2x2 2x2 V)]\nSMMLP[]\n0',
            'P MAX[001(2x2 2x2 V)]\nP MAX[001(2x2 2x2 V)]\nSMMLP[]\n0',
        ]
        self.histories = [
            cga.History(None, None),
            cga.History(None, None),
        ]

        def create(topology):
            return cga.CNN.from_string(topology,
                                       input_shape=self.data.example.shape,
                                       num_classes=self.data.num_classes)

        pickle = {
            'learning_rate_decay_points': [],
            'learning_rate': .1,
            'momentum': .9,
            'learning_rate_decay_factor': None,
            'fitness_punishment_per_hour': 0,
            'cache': {
                self.topologies[0]: {
                    'evaluation': cga.Evaluation(1, 1, 0),
                    'model': create(self.topologies[0])
                },
                self.topologies[1]: {
                    'evaluation': cga.Evaluation(2, 2, 0),
                    'model': create(self.topologies[1])
                },
            }
        }

        self.evaluator = cga.Evaluator.from_pickle(pickle, data=self.data)

    def create_individual(self, topology=None):
        if not topology:
            self.topology_index = 1 - self.topology_index
            topology = self.topologies[self.topology_index]
            history = self.histories[self.topology_index]
        else:
            history = None
        individual = cga.CNN.from_string(topology,
                                         input_shape=self.data.example.shape,
                                         num_classes=self.data.num_classes,
                                         )
        individual.history = history
        return individual

    def test_interface(self):
        search_space = cga.CNN.SearchSpace()

        algorithm = cga.GA(
            self.evaluator,
            self.create_individual,
            population_size=2,
            crossover_probability=1,
            mutation_probability=0,
            epoch_fn=lambda individual, gen_index, prev_evaluations: 0,
        )

        self.assertEqual(algorithm.best_fitness, 2)
        self.assertEqual(algorithm.best_individual.topology,
                         self.topologies[1])

        algorithm.evolve(search_space, elitism=True)
        self.assertEqual(algorithm.best_fitness, 2)

        # check history
        self.assertEqual(
            algorithm.generations[-1][0].history.parent1.history, self.histories[1]
        )
        self.assertEqual(
            algorithm.generations[-1][0].history.parent1, algorithm.best_individual
        )
        self.assertEqual(
            algorithm.generations[-1][0].history.parent2.history, self.histories[1]
        )
        self.assertEqual(
            algorithm.generations[-1][0].history.parent2, algorithm.best_individual
        )

    def test_pickling(self):
        algorithm = cga.GA(
            self.evaluator,
            self.create_individual,
            population_size=2,
            crossover_probability=1,
            mutation_probability=0,
            epoch_fn=lambda individual, gen_index, prev_evaluations: 0
        )

        pickled = algorithm.pickle()
        algorithm2 = cga.GA.from_pickle(
            pickled,
            data=self.data,
            restore_individual=self.create_individual,
            epoch_fn=lambda individual, gen_index, prev_evaluations: 0
        )

        algorithm2.evolve(cga.CNN.SearchSpace(), elitism=True)
        self.assertEqual(algorithm.best_fitness, 2)


if __name__ == '__main__':
    unittest.main()
