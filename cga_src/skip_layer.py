'''Dynamic Convolutional Skip'''

import re
import numpy as np
import torch.nn as nn

from cga_src.convolutional_layer import ConvolutionalLayer
from cga_src.base import DynamicLayer, AbstractSearchSpace, StochasticFunction, LayerContainer


class SkipLayer(DynamicLayer, LayerContainer):
    '''Creates convolutional layer'''

    @staticmethod
    def get_layer_type():
        return 'S'

    class SearchSpace(AbstractSearchSpace):
        '''The skip layer can have a different size of convolutional filters'''

        # Make a copy of the conv default search space so if theirs change ours doesn't
        _default_conv = ConvolutionalLayer.SearchSpace()

        @property
        def default_depth(self):
            '''The initial number of convolutional parts'''
            return (2,)

        @property
        def default_conv(self):
            '''The search space for the encapsulated conv layers'''
            return self._default_conv

        @property
        def default_mutation_func(self):
            '''The default mutations that this layer supports'''

            return StochasticFunction({
                lambda skip_layer: skip_layer.add_conv(self): 0,
                lambda skip_layer: skip_layer.remove_conv(): 0,
                lambda skip_layer: skip_layer.mutate_random_conv(self.conv_search_space): 1,
            })

        def __init__(self, mutate=None, depths=None, conv_search_space=None):
            self.depth = depths or self.default_depth
            self.conv_search_space = conv_search_space or self.default_conv
            self.mutate = mutate or self.default_mutation_func

    @classmethod
    def from_string(cls, string, input_shape):
        # S[002(3x3 1x1);002(3x3 1x1)]

        skip, convs = re.fullmatch(
            r'^(\w+)\{([^\}]+)\}$', string
        ).group(1, 2)

        assert skip == cls.get_layer_type()
        shape = input_shape
        convs = convs.split(';')
        for i, conv in enumerate(convs):
            convs[i] = ConvolutionalLayer.from_string(conv, shape)
            shape = convs[i].output_shape

        return cls(input_shape, convs)

    @classmethod
    def random(cls, input_shape, search_space=None):
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)
        num_convs = np.random.choice(search_space.depth)

        convs = []
        shape = input_shape
        for _ in range(num_convs):
            conv = ConvolutionalLayer.random(shape, search_space=search_space.conv_search_space)
            convs.append(conv)
            shape = conv.output_shape

        return cls(input_shape, convs)

    def __init__(self,
                 input_shape,
                 convs
                 ):
        '''Creates a skip layer according to a string definition'''
        super(SkipLayer, self).__init__()

        assert isinstance(convs, list)

        self._convs = convs
        self._sequential = nn.Sequential(*convs)
        self._adapter = None
        self.change_input_shape(input_shape)

    def clone(self):
        return SkipLayer(input_shape=self._input_shape,
                         convs=[conv.clone() for conv in self._convs]
                         )

    def change_input_shape(self, input_shape):
        # Tell the encapsulated layers that the shape changed
        self._input_shape = input_shape
        shape = input_shape
        for conv in self._convs:
            conv.change_input_shape(shape)
            shape = conv.output_shape

        output_shape = shape
        # SKIP mechanic is only needed when we have more than one conv layer
        # (else it's a simple conv layer)
        if input_shape[0] != output_shape[0] and len(self._convs) > 1:
            # when the output shape changes, the identity needs to be changed appropriately

            #num_filters = output_shape[0]
            #self._adapter = ConvolutionalLayer(input_shape, Filter(num_filters, (1, 1), (1, 1)))

            self._adapter = nn.Conv2d(
                kernel_size=(1, 1),
                stride=(1, 1),
                in_channels=input_shape[0],
                out_channels=output_shape[0],
            )
        else:
            self._adapter = None

    def forward(self, x):
        assert np.all(x.shape[1:] == self._input_shape)

        if self._adapter:
            identity = self._adapter(x)
        else:
            identity = x

        x = self._sequential(x)

        # assert x.shape == identity.shape

        if len(self._convs) > 1:
            # apply the skip function
            x += identity

        return x

    @property
    def topology(self):
        '''The string representation of this topology'''
        return 'S{%s}' % ';'.join([layer.topology for layer in self._convs])

    @property
    def output_shape(self):
        '''The output shape of this layer'''
        return self._convs[-1].output_shape

    def mutate(self, search_space=None):
        '''Mutates one of the conv layers using the stochastic function from the search space'''
        search_space = search_space if search_space else SkipLayer.SearchSpace()
        return search_space.mutate(self)

    def add_conv(self, conv_search_space=None):
        '''Inserts a convolutional layer to a random position'''
        conv_search_space = conv_search_space or self.SearchSpace().conv_search_space
        assert isinstance(conv_search_space, ConvolutionalLayer.SearchSpace)

        new_conv = lambda shape: ConvolutionalLayer.random(shape, search_space=conv_search_space)
        new_convs = self.insert_to_layers(self._convs, self._input_shape, new_conv)

        return SkipLayer(self._input_shape,
                         new_convs
                         )

    def remove_conv(self):
        '''Removes a conv layer at a random position'''
        if len(self._convs) < 3:
            # prevent the skip layer to become a CONV layer or empty
            return self.clone()

        return SkipLayer(self._input_shape,
                         self.remove_from_layers(self._convs, self._input_shape)
                         )

    def mutate_random_conv(self, conv_search_space=None):
        '''Mutates exactly one of the encapsulated convolutional layers'''
        conv_search_space = conv_search_space or self.SearchSpace().conv_search_space
        assert isinstance(conv_search_space, ConvolutionalLayer.SearchSpace)

        mutator = lambda layer, _: layer.mutate(conv_search_space)
        new_convs = self.mutate_in_layers(self._convs, self._input_shape, mutator)

        return SkipLayer(self._input_shape,
                         new_convs
                         )
