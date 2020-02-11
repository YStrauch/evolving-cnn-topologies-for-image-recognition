'''Dynamic Convolutional Layer'''

import re
import numpy as np
import torch.nn as nn

from cga_src.kernel import Kernel
from cga_src.base import DynamicLayer, AbstractSearchSpace


class ConvolutionalLayer(DynamicLayer):
    '''Creates convolutional layer'''

    @staticmethod
    def get_layer_type():
        return 'C'

    class SearchSpace(AbstractSearchSpace):
        '''Defines the searcg space for the pooling layer'''

        @property
        def default_kernel(self):
            '''The default kernel search space'''
            return Kernel.SearchSpace([64, 128, 256], [(3, 3)], [(1, 1)], ['S'])

        def __init__(self, kernel_search_space=None):
            self.kernel_search_space = kernel_search_space or self.default_kernel

    @classmethod
    def from_string(cls, string, input_shape):
        # C[002(3x3 1x1)]

        convolution, kernel = re.fullmatch(
            r'^(\w+)\[([^\]]+)\]$', string
        ).group(1, 2)

        assert convolution == cls.get_layer_type()
        return cls(input_shape, Kernel.from_string(kernel, input_shape))

    @classmethod
    def random(cls, input_shape, search_space=None):
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)

        return cls(input_shape, Kernel.random(
            input_shape,
            search_space.kernel_search_space
        ))

    def __init__(self,
                 input_shape,
                 kernel
                 ):
        '''Creates a pooling layer according to a string definition'''
        super(ConvolutionalLayer, self).__init__()

        assert isinstance(kernel, Kernel)
        self._kernel = kernel
        self._output_shape = None
        self._sequential = None
        self.change_input_shape(input_shape)

    def change_input_shape(self, input_shape):
        self._input_shape = input_shape

        self._kernel.change_input_shape(input_shape)
        self._output_shape, padding = self._kernel.calc_output_shape_and_padding()

        conv = nn.Conv2d(
            kernel_size=tuple(self._kernel.resolution),
            stride=tuple(self._kernel.stride),
            in_channels=int(input_shape[0]),
            out_channels=self._kernel.depth,
            padding=tuple(padding)
        )

        norm = nn.BatchNorm2d(self._kernel.depth)
        relu = nn.ReLU(inplace=True)

        self._sequential = nn.Sequential(conv, relu, norm)

    @property
    def topology(self):
        '''The string representation of this topology'''
        return 'C[%s]' % self._kernel.topology

    @property
    def output_shape(self):
        '''The output shape of this layer'''
        return self._output_shape

    def clone(self):
        return ConvolutionalLayer(input_shape=self._input_shape,
                                  kernel=self._kernel.clone()
                                  )

    def mutate(self, search_space=None):
        '''Mutates this layer'''
        search_space = search_space or self.SearchSpace()
        kernel = self._kernel.mutate(search_space.kernel_search_space)
        return ConvolutionalLayer(self._input_shape, kernel)

    def forward(self, x):
        assert np.all(x.shape[1:] == self._input_shape)

        return self._sequential(x)
