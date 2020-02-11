# pylint: disable=protected-access, missing-docstring

import unittest
import helpers
import cga


class PoolingLayer(helpers.Test):

    def test_serialisation(self):
        self.assertTrue(helpers.serialisation_works(
            cga.PoolingLayer, input_shape=(1, 32, 32)))

    def test_clone(self):
        layer = cga.PoolingLayer.random(input_shape=(1, 32, 32))
        clone = layer.clone()
        self.assertNotEqual(layer, clone)
        self.assertEqual(layer._pooling_type, clone._pooling_type)

    def test_random(self):
        for pooling_type in ('AVG', 'MAX'):
            search_space = cga.PoolingLayer.SearchSpace(
                pooling_types=[pooling_type])
            layer = cga.PoolingLayer.random(input_shape=(1, 32, 32),
                                            search_space=search_space)

            self.assertEqual(layer._pooling_type, pooling_type)

    def test_exception(self):
        with self.assertRaises(cga.CannotReduceResolutionError):
            cga.PoolingLayer.random(input_shape=(1, 13, 13))

        with self.assertRaises(cga.CannotReduceResolutionError):
            cga.PoolingLayer.random(input_shape=(1, 0, 0))

        kernel = cga.Kernel.random([3, 64, 64],
                                   cga.Kernel.SearchSpace([1], [(2, 2)], [(2, 2)], ['V']))

        with self.assertRaises(cga.CannotReduceResolutionError):
            cga.PoolingLayer(input_shape=(1, 3, 3),
                             pooling_type='MAX',
                             kernel=kernel)

    def test_shape(self):
        kernel = cga.Kernel.random([3, 64, 64],
                                   cga.Kernel.SearchSpace([1], [(2, 2)], [(2, 2)], ['V']))

        pool = cga.PoolingLayer(input_shape=[3, 64, 64],
                                pooling_type='MAX', kernel=kernel)
        self.assertArrayEqual(pool.output_shape, [3, 32, 32])

    def test_pooling_type_mutation(self):
        # Flip pooling type
        kernel = cga.Kernel(
            input_shape=(3, 64, 64),
            depth=1,
            resolution=(2, 2),
            stride=(2, 2),
            same_padding=False
        )

        layer = cga.PoolingLayer(
            input_shape=(3, 64, 64),
            pooling_type='MAX',
            kernel=kernel
        )

        search_space = cga.PoolingLayer.SearchSpace(
            pooling_types=['MAX', 'AVG'],
            kernel_search_space=cga.StochasticFunction({
                lambda layer: layer.swap_pooling_type(search_space): 1,
            })
        )
        new_layer = layer.mutate(search_space)
        self.assertNotEqual(new_layer, layer)
        self.assertNotEqual(layer._pooling_type, new_layer._pooling_type)
        self.assertNotEqual(new_layer._kernel, layer._kernel)
        self.assertNotEqual(new_layer.topology, layer.topology)

        double_mutated = new_layer.mutate(search_space)
        self.assertNotEqual(double_mutated, layer)
        self.assertEqual(double_mutated._pooling_type, layer._pooling_type)
        self.assertEqual(double_mutated.topology, layer.topology)

    def test_pooling_kernel_mutation(self):
        # Change pooling kernel
        kernel = cga.Kernel(
            input_shape=(3, 64, 64),
            depth=1,
            resolution=(2, 2),
            stride=(2, 2),
            same_padding=False
        )

        layer = cga.PoolingLayer(
            input_shape=(3, 64, 64),
            pooling_type='MAX',
            kernel=kernel
        )

        kernel_search_space = cga.Kernel.SearchSpace(
            depths=[1],
            resolutions=[(1, 1)],
            strides=[(3, 3)],
            padding_types=['S']
        )

        search_space = cga.PoolingLayer.SearchSpace()
        search_space.mutate = cga.StochasticFunction({
            lambda layer: layer.change_kernel(kernel_search_space): 1,
        })

        new_layer = layer.mutate(search_space)
        self.assertNotEqual(new_layer, layer)
        self.assertEqual(layer._pooling_type, new_layer._pooling_type)
        self.assertNotEqual(new_layer._kernel, layer._kernel)
        self.assertNotEqual(new_layer.topology, layer.topology)
        self.assertArrayEqual(new_layer._kernel.resolution, [1, 1])
        self.assertArrayEqual(new_layer._kernel.stride, [3, 3])


if __name__ == '__main__':
    unittest.main()
