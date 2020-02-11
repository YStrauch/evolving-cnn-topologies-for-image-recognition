# pylint: disable=protected-access, missing-docstring

import unittest
import numpy as np

import helpers
import cga


class SoftmaxMLP(helpers.Test):
    def test_no_hidden(self):
        sm_mlp = cga.SoftmaxMLP(
            input_shape=[3, 32, 32],
            num_classes=10,
            hidden_neurons=[]
        )

        self.assertEqual(sm_mlp.num_classes, 10)
        self.assertEqual(len(sm_mlp._layers), 2)

    def test_one_hidden(self):
        sm_mlp = cga.SoftmaxMLP(
            input_shape=[3, 32, 32],
            num_classes=10,
            hidden_neurons=[20]
        )

        self.assertEqual(sm_mlp.num_classes, 10)
        self.assertEqual(len(sm_mlp._layers), 4)

    def test_random(self):
        sm_mlp = cga.SoftmaxMLP.random(
            input_shape=[3, 32, 32],
            search_space=cga.SoftmaxMLP.SearchSpace(
                hidden_neurons=[(30, 15)]
                # forces it to have two hidden layers (30 and 15 neurons)
            )
        )

        sm_mlp.change_num_classes(10)

        self.assertEqual(sm_mlp.num_classes, 10)
        self.assertEqual(len(sm_mlp._layers), 6)

    def test_serialisation(self):
        self.assertTrue(helpers.serialisation_works(
            cga.SoftmaxMLP, input_shape=(1, 32, 32)
        ))

    def test_mutation(self):
        original = cga.SoftmaxMLP(
            input_shape=[3, 32, 32],
            num_classes=10,
            hidden_neurons=[20]  # force one hidden layer
        )

        search_space = cga.SoftmaxMLP.SearchSpace(
            hidden_neurons=[(20,), (30, 15)]
            # can either have two or one hidden layer
        )

        self.assertEqual(len(original._layers), 4)

        mutated = original.mutate(search_space=search_space)

        # mutation needs to prevent switch from the active state
        self.assertNotEqual(original, mutated)
        self.assertEqual(len(mutated._layers), 6)

        # mutate it again
        double_mutated = mutated.mutate(search_space=search_space)
        # should be like the initial again
        self.assertNotEqual(double_mutated, original)
        self.assertEqual(double_mutated.topology, original.topology)

    def test_clone(self):
        clf1 = cga.SoftmaxMLP.random(input_shape=(3, 32, 32))
        clf2 = clf1.clone()
        self.assertNotEqual(clf1, clf2)
        self.assertEqual(clf1.topology, clf2.topology)


if __name__ == '__main__':
    unittest.main()
