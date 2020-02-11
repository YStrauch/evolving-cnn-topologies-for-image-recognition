import re

import numpy as np
import torch.nn

from cga_src.base import DynamicClassifier, AbstractSearchSpace


class SoftmaxMLP(DynamicClassifier):
    '''A softmax classification MLP'''

    @staticmethod
    def get_layer_type():
        return 'SMMLP'

    class SearchSpace(AbstractSearchSpace):
        '''Defines the parameters and their space'''

        @property
        def default_hidden_neurons(self):
            '''The default hidden layer configuration that this classifier can use'''
            return ([50], [])  # either one hidden layer with 50 neurons or no hidden layer

        def __init__(self,
                     hidden_neurons=None,
                     ):

            self.hidden_neurons = tuple(
                map(list, hidden_neurons or self.default_hidden_neurons)
            )

    def __init__(self,
                 input_shape,
                 hidden_neurons,
                 num_classes=None
                 ):
        super(SoftmaxMLP, self).__init__()

        assert isinstance(hidden_neurons, (list, tuple))
        hidden_neurons = list(hidden_neurons)

        if hidden_neurons:
            assert isinstance(hidden_neurons[0], int)

        self._input_shape = None
        self._flattened_shape = None
        self._num_classes = num_classes
        self._layers = None
        self._sequential = None
        self._hidden_neurons = hidden_neurons

        self.change_input_shape(input_shape)

    def change_input_shape(self, new_input_shape):
        self._input_shape = new_input_shape
        self._flattened_shape = new_input_shape[0] * \
            new_input_shape[1]*new_input_shape[2]
        if self._num_classes is not None:
            self.recalc_layers()

    def change_num_classes(self, new_num_classes):
        self._num_classes = new_num_classes
        if self._input_shape is not None:
            self.recalc_layers()

    def recalc_layers(self):
        self._layers = []
        num_neurons = self._flattened_shape

        for hidden_layer_size in self._hidden_neurons:
            self._layers.append(torch.nn.Linear(
                num_neurons, hidden_layer_size))
            self._layers.append(torch.nn.ReLU())
            num_neurons = hidden_layer_size

        # output layer
        self._layers.append(torch.nn.Linear(num_neurons, self._num_classes))

        # Softmax classification
        self._layers.append(torch.nn.Softmax(dim=1))
        self._sequential = torch.nn.Sequential(*self._layers)

    def clone(self):
        return SoftmaxMLP(self._input_shape, self._hidden_neurons)

    @property
    def topology(self):
        return 'SMMLP[%s]' % (
            ','.join(list(map(str, self._hidden_neurons)))
        )

    @property
    def num_classes(self):
        return self._num_classes

    @classmethod
    def random(cls, input_shape, search_space=None):
        search_space = search_space or SoftmaxMLP.SearchSpace()
        assert isinstance(search_space, SoftmaxMLP.SearchSpace)

        hidden_neurons = search_space.hidden_neurons[
            np.random.randint(0, len(search_space.hidden_neurons))
        ]

        return cls(input_shape=input_shape,
                   hidden_neurons=hidden_neurons
                   )

    @classmethod
    def from_string(cls, string, input_shape):
        hidden_neurons = re.fullmatch(
            r'^SMMLP\[([^\]]*)\]$', string
        ).group(1)
        if hidden_neurons:
            hidden_neurons = hidden_neurons.split(',')
            hidden_neurons = list(map(int, hidden_neurons))
        else:
            hidden_neurons = []

        return cls(input_shape, hidden_neurons)

    def forward(self, x):
        assert np.all(x.shape[1:] == self._input_shape)
        x = x.reshape(-1, self._flattened_shape)
        return self._sequential(x)

    def mutate(self, search_space=None):
        search_space = search_space or SoftmaxMLP.SearchSpace()
        assert isinstance(search_space, SoftmaxMLP.SearchSpace)

        hidden_neurons = self._hidden_neurons
        if len(search_space.hidden_neurons) > 1:
            while hidden_neurons == self._hidden_neurons:
                hidden_neurons = search_space.hidden_neurons[
                    np.random.randint(0, len(search_space.hidden_neurons))
                ]

        return SoftmaxMLP(self._input_shape, hidden_neurons, num_classes=self._num_classes)
