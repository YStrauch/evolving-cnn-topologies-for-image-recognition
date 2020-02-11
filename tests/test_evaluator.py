# pylint: disable=protected-access, missing-docstring

import unittest
import time
import torch
import helpers
import cga


class Evaluator(helpers.Test):
    def setUp(self):
        self.data = cga.MNIST()
        self.cached_m1 = 'C[001(3x3 1x1 S)]\nP MAX[001(2x2 2x2 V)]\nSMMLP[]\n-1'
        self.cached_0 = 'C[001(3x3 1x1 S)]\nP MAX[001(2x2 2x2 V)]\nSMMLP[]\n0'
        self.cached_1 = 'C[001(3x3 1x1 S)]\nP MAX[001(2x2 2x2 V)]\nSMMLP[]\n1'

        def create(str):
            return cga.CNN.from_string(
                str, input_shape=self.data.example.shape, num_classes=10
            )

        self.pickle = {
            'learning_rate_decay_points': [],
            'learning_rate': .1,
            'momentum': .9,
            'learning_rate_decay_factor': None,
            'fitness_punishment_per_hour': 0,
            'cache': {
                self.cached_m1: {
                    'evaluation': cga.Evaluation(100, 0, 123),
                    'model': create(self.cached_m1)
                },
                self.cached_0: {
                    'evaluation': cga.Evaluation(101, 0, 123),
                    'model': create(self.cached_0)
                }
            }
        }
        self.evaluator = cga.Evaluator.from_pickle(self.pickle, data=self.data)

    def test_cuda(self):
        self.assertTrue(torch.cuda.is_available())

    def test_interface_and_cache(self):
        start_time = time.time()
        evaluation1 = self.evaluator.get_evaluation(self.cached_m1).result()
        evaluation2 = self.evaluator.get_evaluation(self.cached_0).result()
        self.assertLess(time.time() - start_time, 1)
        self.assertEqual(evaluation1.fitness, 100)
        self.assertEqual(evaluation2.fitness, 101)

    def test_cache_future(self):
        # Assert that the evaluator caches futures
        topology = 'C[1(3x3 1x1 S)]\nSMMLP[]\n-1'
        topologies = [topology for i in range(100)]
        start_time = time.time()
        results = list(self.evaluator.get_evaluations(topologies))
        self.assertLess(time.time() - start_time, 10)
        self.assertEqual(results[0][0], topology)

    def test_regularisation(self):
        evaluator = cga.Evaluator.from_pickle(self.pickle, data=self.data)
        evaluator.fitness_punishment_per_hour = 1
        topology = 'C[1(3x3 1x1 S)]\nSMMLP[]\n0'
        evaluation = evaluator.get_evaluation(topology).result()

        self.assertLess(evaluation.fitness, evaluation.accuracy)

    def test_pickling(self):
        evaluator = cga.Evaluator(data=self.data)
        topology = 'C[1(3x3 1x1 S)]\nSMMLP[]\n0'
        evaluation1 = evaluator.get_evaluation(topology).result()
        evaluator = cga.Evaluator.from_pickle(
            evaluator.pickle(), data=self.data)

        start_time = time.time()
        evaluation2 = evaluator.get_evaluation(topology).result()

        self.assertLess(time.time() - start_time, 1)
        self.assertEqual(evaluation1.fitness, evaluation2.fitness)

    def test_partial_training(self):
        evaluator = cga.Evaluator.from_pickle(self.pickle, data=self.data)
        start_time = time.time()
        evaluation = evaluator.get_evaluation(self.cached_1).result()
        duration = time.time() - start_time
        self.assertGreater(duration, 3)
        self.assertGreater(evaluation.fitness, 0)
        self.assertLess(evaluation.fitness, 1)

        self.assertEqual(
            evaluator.cache[self.cached_m1]['model'].topology, self.cached_m1
        )
        self.assertEqual(
            evaluator.cache[self.cached_0]['model'].topology, self.cached_0
        )
        self.assertEqual(
            evaluator.cache[self.cached_1]['model'].topology, self.cached_1
        )


if __name__ == '__main__':
    unittest.main()
