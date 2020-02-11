# pylint: disable=protected-access, missing-docstring

import unittest
import time
import helpers
import cga


class Population(helpers.Test):
    def setUp(self):
        self.better = cga.CNN.from_string(
            'C[001(3x3 1x1 S)]\nSMMLP[]\n0', input_shape=(1, 32, 32), num_classes=10
        )
        self.worse = cga.CNN.from_string(
            'C[002(3x3 1x1 S)]\nSMMLP[]\n0', input_shape=(1, 32, 32), num_classes=10
        )

        data = cga.MNIST()

        pickle = {
            'learning_rate_decay_points': [],
            'learning_rate': .1,
            'momentum': .9,
            'learning_rate_decay_factor': None,
            'fitness_punishment_per_hour': 0,
            'cache': {
                self.worse.topology: {
                    'evaluation': cga.Evaluation(0, 0, 0),
                    'model': None
                },
            }
        }

        self.evaluator = cga.Evaluator.from_pickle(pickle, data=data)

    def test_interface(self):
        # adding and evaluating the worse individual must be resolved
        # instantly because it is in the cache
        start_time = time.time()
        pop = cga.Population(self.evaluator,
                             individuals=[self.worse])
        self.assertEqual(pop.worst_fitness, 0)
        self.assertEqual(pop.best_fitness, 0)
        self.assertLess(time.time() - start_time, 1)

        # adding the other one must also happen instantly because we
        # are not evaluating it
        start_time = time.time()
        pop.append(self.better)
        self.assertLess(time.time() - start_time, 1)

        # querying an accuracy should actually start calculating it
        start_time = time.time()
        fitness = pop.best_fitness
        self.assertGreater(time.time() - start_time, 1)

        self.assertEqual(pop.best_fitness, fitness)
        self.assertEqual(pop.worst_fitness, 0)

        self.assertEqual(pop.best_individual, self.better)
        self.assertEqual(pop.worst_individual, self.worse)

        # test tournament selection
        self.assertEqual(pop.tournament_selection(), self.better)
        self.assertEqual(pop.tournament_selection(
            select_worse=True), self.worse)

    def test_add(self):
        pop = cga.Population(self.evaluator)
        pop += ['foo', 'bar']

        assert len(pop) == 2


if __name__ == '__main__':
    unittest.main()
