# pylint: disable=protected-access, missing-docstring

import unittest
import helpers
import cga


class ConvolutionalLayer(helpers.Test):

    def test_serialisation(self):
        self.assertTrue(helpers.serialisation_works(cga.ConvolutionalLayer,
                                                    input_shape=(1, 32, 32)))

    def test_clone(self):
        layer = cga.ConvolutionalLayer.random(input_shape=(1, 2, 3))
        clone = layer.clone()
        self.assertNotEqual(layer, clone)
        kernel1 = layer._kernel
        kernel2 = clone._kernel
        self.assertNotEqual(kernel1, kernel2)
        self.assertArrayEqual(kernel1.resolution, kernel2.resolution)
        self.assertArrayEqual(kernel1.stride, kernel2.stride)
        self.assertEqual(kernel1.depth, kernel2.depth)

    def test_mutation(self):
        # TODO: kernel has no search space and always reinitialises randomly
        layer = cga.ConvolutionalLayer.random((1, 64, 64))
        new_layer = layer.mutate()
        self.assertNotEqual(new_layer, layer)
        self.assertNotEqual(new_layer._kernel, layer._kernel)


if __name__ == '__main__':
    unittest.main()
