# pylint: disable=protected-access, missing-docstring

import unittest
import numpy as np

import helpers
import cga


class CNN(helpers.Test):
    def setUp(self):
        self.input_shape = [1, 32, 32]
        self.history1 = cga.History(None, None)
        self.history2 = cga.History(None, None)

        self.search_space_filter = cga.Kernel.SearchSpace(
            depths=[1],
            resolutions=[(2, 2)],
            strides=[(2, 2)],
            padding_types=('S',)
        )

        self.search_space_conv = cga.ConvolutionalLayer.SearchSpace(
            kernel_search_space=self.search_space_filter
        )

        self.search_space_pool = cga.PoolingLayer.SearchSpace(
            pooling_types=['MAX'],
            kernel_search_space=self.search_space_filter
        )

        self.search_space_skip = cga.SkipLayer.SearchSpace(
            depths=[3],
            conv_search_space=self.search_space_conv
        )

        self.search_space_cnn_force_pool = cga.CNN.SearchSpace(
            layers=[cga.PoolingLayer],
            init_depth_range=[1, 1],
            conv_search_space=self.search_space_conv,
            skip_search_space=self.search_space_skip,
            pool_search_space=self.search_space_pool
        )

        self.search_space_cnn_force_skip = cga.CNN.SearchSpace(
            layers=[cga.SkipLayer],
            init_depth_range=[1, 1],
            conv_search_space=self.search_space_conv,
            skip_search_space=self.search_space_skip,
            pool_search_space=self.search_space_pool
        )

        self.search_space_cnn_force_conv = cga.CNN.SearchSpace(
            layers=[cga.ConvolutionalLayer],
            init_depth_range=[1, 1],
            conv_search_space=self.search_space_conv,
            skip_search_space=self.search_space_skip,
            pool_search_space=self.search_space_pool
        )

    def test_serialisation(self):
        self.assertTrue(helpers.serialisation_works(
            cga.CNN, input_shape=(1, 32, 32), num_classes=10))

    def test_clone(self):
        cnn = cga.CNN.random(input_shape=(1, 32, 32),
                             num_classes=10)

        clone = cnn.clone()
        self.assertNotEqual(cnn, clone)
        self.assertEqual(clone.history.parent1, cnn)

    def test_random(self):
        cnn1 = cga.CNN.random(input_shape=self.input_shape,
                              num_classes=10,
                              search_space=self.search_space_cnn_force_conv)
        cnn2 = cga.CNN.random(input_shape=self.input_shape,
                              num_classes=10,
                              search_space=self.search_space_cnn_force_pool)

        self.assertNotEqual(cnn1, cnn2)
        self.assertNotEqual(cnn1.topology, cnn2.topology)
        self.assertEqual(len(cnn1._layers), len(cnn2._layers))
        self.assertEqual(len(cnn1._layers), 1)
        self.assertIsInstance(cnn1._layers[0], cga.ConvolutionalLayer)
        self.assertIsInstance(cnn2._layers[0], cga.PoolingLayer)

    def test_exception(self):
        with self.assertRaises(cga.CannotReduceResolutionError):
            cga.CNN.from_string('P MAX[001(2x2 2x2 V)]\nSMMLP[]\n0',
                                input_shape=(1, 3, 3),
                                num_classes=10,
                                )

    def test_mutate_pool(self):
        cnn1 = cga.CNN.random(input_shape=self.input_shape,
                              num_classes=10,
                              search_space=self.search_space_cnn_force_pool)

        cnn1.history = self.history1

        # force the pool to mutate
        search_space = cga.CNN.SearchSpace()
        search_space.mutate = cga.StochasticFunction({
            lambda cnn: cnn.change_layer(search_space):  1
        })

        cnn2 = cnn1.mutate(search_space)

        self.assertNotEqual(cnn1, cnn2)
        self.assertNotEqual(cnn1.topology, cnn2.topology)
        self.assertEqual(len(cnn2._layers), len(cnn2._layers))
        self.assertNotEqual(
            cnn1._layers[0]._pooling_type, cnn2._layers[0]._pooling_type)
        self.assertEqual(cnn1.num_classes, cnn2.num_classes)
        self.assertEqual(cnn2.history.parent1, cnn1)
        self.assertEqual(cnn2.history.parent1.history, self.history1)

    def test_mutate_conv(self):
        cnn1 = cga.CNN.random(input_shape=self.input_shape,
                              num_classes=10,
                              search_space=self.search_space_cnn_force_conv)

        cnn1.history = self.history1

        # force the conv to mutate
        search_space = cga.CNN.SearchSpace()
        search_space.dynamic_child_search_spaces[cga.ConvolutionalLayer].kernel_search_space = cga.Kernel.SearchSpace(
            depths=[64],
            resolutions=[(3, 3)],
            strides=[(1, 1)],
            padding_types=('V',)
        )
        search_space.mutate = cga.StochasticFunction({
            lambda cnn: cnn.change_layer(search_space):  1
        })

        cnn2 = cnn1.mutate(search_space)

        self.assertNotEqual(cnn1, cnn2)
        self.assertNotEqual(cnn1.topology, cnn2.topology)
        self.assertEqual(len(cnn2._layers), len(cnn2._layers))
        self.assertNotEqual(
            cnn1._layers[0]._kernel.depth, cnn2._layers[0]._kernel.depth)
        self.assertEqual(cnn2._layers[0]._kernel.depth, 64)
        self.assertEqual(cnn1.num_classes, cnn2.num_classes)
        self.assertEqual(cnn2.history.parent1, cnn1)
        self.assertEqual(cnn2.history.parent1.history, self.history1)

    def test_insert_conv(self):
        cnn1 = cga.CNN.random(input_shape=self.input_shape,
                              num_classes=10,
                              search_space=self.search_space_cnn_force_pool)

        cnn1.history = self.history1

        # force a new conv layer
        conv_search_space = cga.ConvolutionalLayer.SearchSpace()
        conv_search_space.kernel_search_space = cga.Kernel.SearchSpace(
            depths=[64],
            resolutions=[(3, 3)],
            strides=[(1, 1)],
            padding_types=('V',)
        )

        cnn_search_space = cga.CNN.SearchSpace()
        cnn_search_space.mutate = cga.StochasticFunction({
            lambda cnn: cnn.insert_layer(conv_search_space, cga.ConvolutionalLayer):  1
        })

        cnn_search_space.dynamic_child_search_spaces[cga.ConvolutionalLayer] = conv_search_space
        cnn2 = cnn1.mutate(cnn_search_space)

        self.assertNotEqual(cnn1, cnn2)
        self.assertNotEqual(cnn1.topology, cnn2.topology)
        self.assertEqual(len(cnn2._layers), 2)
        conv_exists = isinstance(cnn2._layers[0], cga.ConvolutionalLayer) or isinstance(
            cnn2._layers[1], cga.ConvolutionalLayer)
        self.assertTrue(conv_exists)
        self.assertEqual(cnn2.history.parent1, cnn1)
        self.assertEqual(cnn2.history.parent1.history, self.history1)

    def test_shape_propagation(self):
        # Create a CNN with a CONV and a POOL layer
        input_shape = np.array([1, 32, 32])

        cnn = cga.CNN(
            input_shape=input_shape,
            num_classes=10,
            layers=[
                cga.ConvolutionalLayer(
                    input_shape=input_shape,
                    kernel=cga.Kernel(
                        input_shape,
                        depth=4,
                        resolution=(3, 3),
                        stride=(1, 1),
                        same_padding=True
                    )
                ),
                cga.PoolingLayer(
                    input_shape=np.array([4, 32, 32]),
                    pooling_type='AVG',
                    kernel=cga.Kernel(
                        np.array([4, 32, 32]),
                        depth=1,
                        resolution=(2, 2),
                        stride=(2, 2),
                        same_padding=False
                    )
                )
            ],
            clf=cga.SoftmaxMLP(
                input_shape=[4, 16, 16],
                hidden_neurons=[],
                num_classes=10
            ),
            num_epochs=1,
            history=cga.History(None, 'test')
        )

        self.assertArrayEqual(cnn._layers[0]._input_shape, input_shape)
        self.assertArrayEqual(cnn._layers[0].output_shape, [4, 32, 32])
        self.assertArrayEqual(cnn._layers[1]._input_shape, [4, 32, 32])
        self.assertArrayEqual(cnn._layers[1].output_shape, [4, 16, 16])
        self.assertArrayEqual(cnn._clf._input_shape, [4, 16, 16])
        self.assertEqual(cnn._clf.num_classes, 10)

        # Then prepend a new pooling layer to index 0
        cnn = cnn.insert_layer(
            Layer=cga.PoolingLayer,
            index=0,
            search_space=cga.PoolingLayer.SearchSpace(
                pooling_types=['AVG'],
                kernel_search_space=cga.Kernel.SearchSpace(
                    depths=[1],
                    resolutions=[(2, 2)],
                    strides=[(2, 2)],
                    padding_types=['V']
                )
            )
        )

        # The new layer needs to have the correct shape
        self.assertArrayEqual(cnn._layers[0]._input_shape, input_shape)
        self.assertArrayEqual(cnn._layers[0].output_shape, [1, 16, 16])
        self.assertArrayEqual(cnn._layers[1]._input_shape, [1, 16, 16])
        self.assertArrayEqual(cnn._layers[1].output_shape, [4, 16, 16])
        self.assertArrayEqual(cnn._layers[2]._input_shape, [4, 16, 16])
        self.assertArrayEqual(cnn._layers[2].output_shape, [4, 8, 8])
        self.assertArrayEqual(cnn._clf._input_shape, [4, 8, 8])
        self.assertEqual(cnn._clf.num_classes, 10)

    def test_crossover(self):
        cnn1 = cga.CNN.from_string('S{C[064(3x3 1x1 S)];C[128(3x3 1x1 S)]}\nS{C[032(3x3 1x1 S)];C[128(3x3 1x1 S)]}\nSMMLP[]\n1',
                                   input_shape=(1, 32, 32),
                                   num_classes=10,
                                   )
        cnn1.history = self.history1

        cnn2 = cga.CNN.from_string('P MAX[001(2x2 2x2 V)]\nP AVG[001(2x2 2x2 V)]\nSMMLP[15,10]\n1',
                                   input_shape=(1, 32, 32),
                                   num_classes=10,
                                   )
        cnn2.history = self.history2
        # assert correct output shapes
        self.assertArrayEqual(cnn1._layers[0].output_shape, [128, 32, 32])
        self.assertArrayEqual(cnn1._layers[1].output_shape, [128, 32, 32])

        self.assertArrayEqual(cnn2._layers[0].output_shape, [1, 16, 16])
        self.assertArrayEqual(cnn2._layers[1].output_shape, [1, 8, 8])

        children = cnn1.one_point_crossover(cnn2)

        self.assertNotEqual(cnn1, cnn2)
        self.assertNotEqual(children[0], children[1])
        self.assertNotEqual(children[0], cnn1)
        self.assertNotEqual(children[1], cnn2)
        self.assertNotEqual(children[0].topology, children[1].topology)

        # we can have exactly two offsprings, however we don't know which one is which
        expected_topologies = [
            'S{C[064(3x3 1x1 S)];C[128(3x3 1x1 S)]}\nP AVG[001(2x2 2x2 V)]\nSMMLP[15,10]\n1',
            'P MAX[001(2x2 2x2 V)]\nS{C[032(3x3 1x1 S)];C[128(3x3 1x1 S)]}\nSMMLP[]\n1'
        ]

        # so we need to find it out and sort the children
        if children[0].topology != expected_topologies[0]:
            children = list(reversed(children))

        self.assertEqual(children[0].topology, expected_topologies[0])
        self.assertEqual(children[1].topology, expected_topologies[1])

        # assert correct output shapes
        self.assertArrayEqual(
            children[0]._layers[0].output_shape, [128, 32, 32]
        )
        self.assertArrayEqual(
            children[0]._layers[1].output_shape, [128, 16, 16]
        )

        self.assertArrayEqual(
            children[1]._layers[0].output_shape, [1, 16, 16]
        )
        self.assertArrayEqual(
            children[1]._layers[1].output_shape, [128, 16, 16]
        )

        # assert history
        self.assertEqual(children[0].history.parent1, cnn1)
        self.assertEqual(children[0].history.parent2, cnn2)
        self.assertEqual(children[0].history.parent1.history, self.history1)
        self.assertEqual(children[0].history.parent2.history, self.history2)

        self.assertEqual(children[1].history.parent1, cnn2)
        self.assertEqual(children[1].history.parent2, cnn1)
        self.assertEqual(children[1].history.parent1.history, self.history2)
        self.assertEqual(children[1].history.parent2.history, self.history1)

        self.assertEqual(children[1].history.cross_over_point1, 1)
        self.assertEqual(children[1].history.cross_over_point2, 1)


if __name__ == '__main__':
    unittest.main()
