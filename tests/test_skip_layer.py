# pylint: disable=protected-access, missing-docstring

import unittest
import numpy as np

import helpers
import cga


class SkipLayer(helpers.Test):
    def setUp(self):
        # Forces a deterministic kernel so all mutations
        # that influence the kernel can be checked easily
        self.conv_search_space = cga.ConvolutionalLayer.SearchSpace(
            kernel_search_space=cga.Kernel.SearchSpace(
                resolutions=[(3, 3)],
                strides=[(1, 1)],
                depths=[888],
                padding_types=['S']
            )
        )

    def test_serialisation(self):
        self.assertTrue(helpers.serialisation_works(
            cga.SkipLayer, input_shape=(1, 32, 32)))

    def test_from_string(self):
        layer = cga.SkipLayer.from_string(
            'S{C[064(3x3 1x1 S)];C[128(3x3 1x1 S)]}', input_shape=[1, 28, 28])
        self.assertArrayEqual(layer.output_shape, [128, 28, 28])

    def test_random(self):
        search_space = cga.SkipLayer.SearchSpace(
            depths=(10,)
        )
        skip = cga.SkipLayer.random([1, 1, 1], search_space=search_space)
        self.assertEqual(len(skip._convs), 10)

    def test_clone(self):
        search_space = cga.SkipLayer.SearchSpace(
            depths=(2,)
        )
        layer = cga.SkipLayer.random([1, 1, 1], search_space=search_space)
        self.assertNotEqual(layer, layer.clone())
        conv1 = layer._convs[0]
        conv2 = layer._convs[1]
        self.assertNotEqual(conv1, conv2)

    def test_adapter(self):
        layer = cga.SkipLayer.from_string(
            'S{C[064(3x3 1x1 S)];C[128(3x3 1x1 S)]}',
            input_shape=[1, 32, 32]
        )
        self.assertArrayEqual(layer.output_shape, [128, 32, 32])
        self.assertIsNotNone(layer._adapter)

    def test_mutate_change_layer(self):
        layer = cga.SkipLayer.from_string(
            'S{C[064(3x3 1x1 S)]}',
            input_shape=[1, 32, 32]
        )

        # force the change random conv mutation
        search_space = cga.SkipLayer.SearchSpace()
        search_space.mutate = cga.StochasticFunction({
            lambda layer: layer.mutate_random_conv(self.conv_search_space): 1
        })

        new_layer = layer.mutate(search_space)

        self.assertNotEqual(new_layer, layer)
        self.assertEqual(len(layer._convs), len(new_layer._convs))
        self.assertNotEqual(layer._convs[0], new_layer._convs[0])
        self.assertArrayEqual(new_layer.output_shape, [888, 32, 32])

    def test_mutate_add_layer(self):
        layer = cga.SkipLayer([1, 32, 32], [])

        # force the add conv mutation
        search_space = cga.SkipLayer.SearchSpace()
        search_space.mutate = cga.StochasticFunction({
            lambda layer: layer.add_conv(self.conv_search_space): 1
        })

        new_layer = layer.mutate(search_space)
        self.assertNotEqual(new_layer, layer)
        self.assertNotEqual(len(layer._convs), len(new_layer._convs))
        self.assertArrayEqual(new_layer.output_shape, [888, 32, 32])

    def test_mutate_remove_layer(self):
        layer = cga.SkipLayer.from_string(
            'S{C[064(3x3 1x1 S)];C[032(3x3 1x1 S)];C[128(3x3 1x1 S)]}',
            input_shape=[1, 32, 32]
        )

        # force the remove conv mutation
        search_space = cga.SkipLayer.SearchSpace()
        search_space.mutate = cga.StochasticFunction({
            lambda layer: layer.remove_conv(): 1
        })

        new_layer = layer.mutate(search_space)

        self.assertNotEqual(new_layer, layer)
        self.assertEqual(len(new_layer._convs), 2)

    def test_shape_propagation(self):
        # Create a skip layer with one CONV layer
        input_shape = np.array([1, 32, 32])

        layer = cga.SkipLayer(
            input_shape=input_shape,
            convs=[
                cga.ConvolutionalLayer(
                    input_shape=input_shape,
                    kernel=cga.Kernel(
                        input_shape,
                        depth=4,
                        resolution=(3, 3),
                        stride=(1, 1),
                        same_padding=True
                    )
                )
            ]
        )

        self.assertArrayEqual(layer._convs[0]._input_shape, input_shape)
        self.assertArrayEqual(layer._convs[0].output_shape, [4, 32, 32])
        self.assertArrayEqual(layer.output_shape, [4, 32, 32])

        def new_conv(shape):
            return cga.ConvolutionalLayer(
                input_shape=shape,
                kernel=cga.Kernel(
                    input_shape=shape,
                    depth=16,
                    resolution=[3, 3],
                    stride=[1, 1],
                    same_padding=True
                )
            )
        new_convs = layer.insert_to_layers(
            layer._convs, layer._input_shape, new_conv, index=0
        )

        # The new layer needs to have the correct shape
        self.assertArrayEqual(new_convs[0]._input_shape, input_shape)
        self.assertArrayEqual(new_convs[0].output_shape, [16, 32, 32])
        self.assertArrayEqual(new_convs[1]._input_shape, [16, 32, 32])
        self.assertArrayEqual(new_convs[1].output_shape, [4, 32, 32])

        # And the new layer needs to adapt too
        layer = cga.SkipLayer(input_shape=input_shape, convs=new_convs)
        self.assertArrayEqual(layer.output_shape, [4, 32, 32])


if __name__ == '__main__':
    unittest.main()
