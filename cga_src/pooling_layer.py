'''The dynamic pooling layer'''

import re

import numpy as np
import torch.nn as nn

from cga_src.base import DynamicLayer, AbstractSearchSpace, StochasticFunction
from cga_src.kernel import Kernel


class PoolingLayer(DynamicLayer):
    '''Creates pooling layer'''

    @staticmethod
    def get_layer_type():
        return 'P'

    class SearchSpace(AbstractSearchSpace):
        '''Defines the searcg space for the pooling layer'''
        @property
        def implemented_pooling_types(self):
            '''These pooling layers are supported'''
            return ['AVG', 'MAX']

        @property
        def default_kernel(self):
            '''The default kernel search space'''
            return Kernel.SearchSpace([None], [(2, 2)], [(2, 2)], ['V'])

        @property
        def default_mutation_func(self):
            '''The default mutations that this layer supports'''

            return StochasticFunction({
                lambda pooling_layer: pooling_layer.swap_pooling_type(self): 1,
                lambda pooling_layer: pooling_layer.change_kernel(self.kernel_search_space): 0,
            })

        def __init__(self, pooling_types=None, kernel_search_space=None, mutate=None):
            self.pooling_types = pooling_types or self.implemented_pooling_types
            self.kernel_search_space = kernel_search_space or self.default_kernel
            self.mutate = mutate or self.default_mutation_func

    @classmethod
    def from_string(cls, string, input_shape):
        # P MAX[(2x2 2x2)]

        pool, pooling_type, kernel = re.fullmatch(
            r'^(\w+) (\w+)\[([^\]]+)\]$', string
        ).group(1, 2, 3)

        assert pool == cls.get_layer_type()
        return cls(input_shape, pooling_type, Kernel.from_string(kernel, input_shape))

    @classmethod
    def random(cls, input_shape, search_space=None):
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)
        pooling_type = np.random.choice(search_space.pooling_types)

        return cls(input_shape,
                   pooling_type,
                   Kernel.random(input_shape, search_space.kernel_search_space)
                   )

    def __init__(self,
                 input_shape,
                 pooling_type,
                 kernel
                 ):
        '''Creates a pooling layer according to a string definition'''
        super(PoolingLayer, self).__init__()

        assert isinstance(kernel, Kernel)
        assert pooling_type in PoolingLayer.SearchSpace().implemented_pooling_types

        self._pooling_type = pooling_type
        self._kernel = kernel
        self._output_shape = None
        self.change_input_shape(input_shape)

    def change_input_shape(self, input_shape):
        self._kernel.change_input_shape(input_shape)
        self._output_shape, padding = self._kernel.calc_output_shape_and_padding()
        # pooling layers don't change the z layer
        self._input_shape = input_shape
        self._output_shape[0] = input_shape[0]

        if self._pooling_type == 'MAX':
            self._layer = nn.MaxPool2d(
                kernel_size=tuple(self._kernel.resolution),
                stride=tuple(self._kernel.stride),
                padding=tuple(padding)
            )
        elif self._pooling_type == 'AVG':
            self._layer = nn.AvgPool2d(
                kernel_size=tuple(self._kernel.resolution),
                stride=tuple(self._kernel.stride),
                padding=tuple(padding)
            )
        else:
            raise NotImplementedError(
                'Pooling type %s is not implemented' % self._pooling_type)

    def clone(self):
        return PoolingLayer(input_shape=self._input_shape,
                            pooling_type=self._pooling_type,
                            kernel=self._kernel.clone()
                            )

    @property
    def topology(self):
        '''The string representation of this topology'''
        return 'P %s[%s]' % (self._pooling_type, self._kernel.topology)

    @property
    def output_shape(self):
        '''The output shape of this layer'''
        return self._output_shape

    def forward(self, x):
        assert np.all(x.shape[1:] == self._input_shape)

        return self._layer(x)

    def mutate(self, search_space=None):
        '''Mutates this pooling layer using the StochasticFunction from the SearchSpace'''
        search_space = search_space if search_space else self.SearchSpace()
        return search_space.mutate(self)

    def swap_pooling_type(self, search_space=None):
        '''Swaps the pooling type'''
        assert isinstance(search_space, self.SearchSpace)

        # ensure that we will not mutate to the same pooling type
        search_space = search_space or self.SearchSpace()

        possible_types = search_space.pooling_types.copy()
        del possible_types[possible_types.index(self._pooling_type)]
        assert possible_types

        new_pooling_type = np.random.choice(possible_types)

        return PoolingLayer(input_shape=self._input_shape,
                            pooling_type=new_pooling_type,
                            kernel=self._kernel.clone()
                            )

    def change_kernel(self, kernel_search_space=None):
        '''Changes the kernel'''
        search_space = kernel_search_space or self.SearchSpace().kernel_search_space
        assert isinstance(kernel_search_space, Kernel.SearchSpace)

        new_kernel = self._kernel.mutate(search_space)

        return PoolingLayer(input_shape=self._input_shape,
                            pooling_type=self._pooling_type,
                            kernel=new_kernel
                            )
