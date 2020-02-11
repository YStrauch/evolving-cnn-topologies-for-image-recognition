# pylint: disable=protected-access, missing-docstring

import unittest
import helpers
import cga


class Data(helpers.Test):

    def test_mnist(self):
        mnist = cga.MNIST()
        self.assertTensorShape(mnist.example, [1, 32, 32])

    def test_cifar10(self):
        cifar = cga.CIFAR10()
        self.assertTensorShape(cifar.example, [3, 32, 32])


if __name__ == '__main__':
    unittest.main()
