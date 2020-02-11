'''Skip and Pooling layers'''

import re
import numpy as np
import torch

from cga_src.base import CannotReduceResolutionError, DynamicLayer, LayerContainer,\
    AbstractSearchSpace, StochasticFunction, DynamicClassifier
from cga_src.skip_layer import SkipLayer
from cga_src.pooling_layer import PoolingLayer
from cga_src.convolutional_layer import ConvolutionalLayer
from cga_src.softmax_mlp import SoftmaxMLP


class History():
    def __init__(self, parent1, reason, parent2=None, cross_over_point1=None, cross_over_point2=None):
        self.parent1 = parent1
        self.parent2 = parent2
        self.cross_over_point1 = cross_over_point1
        self.cross_over_point2 = cross_over_point2
        self.reason = reason

        # will be filled out in the pickling process
        self.evaluation = None


class CNN(LayerContainer):
    '''Creates a CNN according to the topology specification'''

    class SearchSpace(AbstractSearchSpace):
        '''Defines the parameters and their space'''
        @property
        def default_layers(self):
            '''The default layers that this CNN uses'''
            return [
                SkipLayer,
                PoolingLayer,
                ConvolutionalLayer,
            ]

        @property
        def default_init_range(self):
            '''The default range of layers that this CNN initialises with'''
            return (2, 10)

        @property
        def default_clfs(self):
            '''The default classifiers that this CNN uses'''
            return [
                SoftmaxMLP
            ]

        @property
        def default_mutation_func(self):
            '''The default mutations that this CNN uses'''

            return StochasticFunction({
                lambda cnn: cnn.insert_layer(
                    self.dynamic_child_search_spaces[SkipLayer], SkipLayer
                ): .7,
                lambda cnn: cnn.insert_layer(
                    self.dynamic_child_search_spaces[PoolingLayer], PoolingLayer
                ): .1,
                lambda cnn: cnn.insert_layer(
                    self.dynamic_child_search_spaces[ConvolutionalLayer], ConvolutionalLayer
                ):  0,
                lambda cnn: cnn.remove_layer(self):  .1,
                lambda cnn: cnn.change_layer(self):  .1,
                lambda cnn: cnn.mutate_classifier(self):  0,
                lambda cnn: cnn.mutate_training_epochs(self): 0
            })

        @property
        def default_num_epoch_range(self):
            '''The default range of training epochs'''
            return (10, 10)

        def default_num_epoch_mutation(self, previous_num_epochs):
            sign = np.random.choice([1, -1])
            # low is inclusive, high exclusive, so we add either: -2, -1, +1, +2
            return previous_num_epochs + sign * np.random.randint(1, 3)

        def __init__(self,
                     layers=None,
                     clfs=None,
                     init_depth_range=None,
                     num_epoch_range=None,
                     num_epoch_mutation=None,
                     mutate=None,
                     conv_search_space=None,
                     skip_search_space=None,
                     pool_search_space=None,
                     clf_search_space=None
                     ):
            self.layers = layers or self.default_layers
            self.initialisation_range = init_depth_range or self.default_init_range
            self.mutate = mutate or self.default_mutation_func
            self.num_epoch_range = num_epoch_range or self.default_num_epoch_range
            self.num_epoch_mutation = num_epoch_mutation or self.default_num_epoch_mutation

            self.clfs = clfs or self.default_clfs

            self.dynamic_child_search_spaces = {
                # layers
                ConvolutionalLayer: conv_search_space or ConvolutionalLayer.SearchSpace(),
                PoolingLayer: pool_search_space or PoolingLayer.SearchSpace(),
                SkipLayer: skip_search_space or SkipLayer.SearchSpace(),

                # classifier
                SoftmaxMLP: clf_search_space
            }

    @classmethod
    def from_string(cls, string, input_shape, num_classes):
        '''Create a CNN using a topology string'''
        # Remove tabs and doubled white spaces from the topology
        pattern = re.compile(r'\t|  +')
        string = re.sub(pattern, '', string).strip()

        layers = string.split('\n')
        shape = input_shape

        for i, layer in enumerate(layers[:-1]):
            layer_type = re.match(
                r'(^\w+)', layer
            ).group(1)

            layer = CNN._init_layer(layer_type, shape, layer)

            if isinstance(layer, DynamicClassifier):
                # classifiers need to know their number of classes
                layer.change_num_classes(num_classes)
            else:
                shape = layer.output_shape

            layers[i] = layer

        # The second last layer is a classifier
        clf = layers[-2]
        assert isinstance(
            clf, DynamicClassifier), 'Last layer needs to be a classifier'

        # The last entry is the number of epochs
        num_epochs = int(layers[-1])

        layers = layers[:-2]

        return cls(input_shape,
                   num_classes,
                   layers,
                   clf,
                   num_epochs,
                   history=History(None, 'from_string')
                   )

    @staticmethod
    def _init_layer(layer_type, shape, string):
        # pylint: disable=invalid-name
        for CurrentLayer in CNN.SearchSpace().layers:
            if CurrentLayer.get_layer_type() == layer_type:
                return CurrentLayer.from_string(string, shape)

        for CurrentClf in CNN.SearchSpace().clfs:
            if CurrentClf.get_layer_type() == layer_type:
                return CurrentClf.from_string(string, shape)

        raise NotImplementedError(
            'CNN Layer type %s not implemented or configured' % layer_type)

    @classmethod
    def random_approach3(cls,
                         input_shape,
                         num_classes,
                         search_space=None):
        '''Returns a random CNN'''
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)

        depth_of_layers = np.random.randint(search_space.initialisation_range[0],
                                            search_space.initialisation_range[1]+1
                                            )

        number_of_convs = min(depth_of_layers/2, np.log2(input_shape[2]))

        # shape = np.asarray(input_shape, dtype='float')
        shape = input_shape
        layers = []
        while len(layers) < depth_of_layers:
            try:
                shape, layer = cls._create_random_layer(input_shape=shape,
                                                        search_space=search_space,
                                                        pooling_prob=1/number_of_convs
                                                        )
            except CannotReduceResolutionError:
                continue
            layers += [layer]

        num_epochs = np.random.randint(search_space.num_epoch_range[0],
                                       search_space.num_epoch_range[1]+1
                                       )

        # Add a random classifier
        clf = cls._create_random_classifier(shape, num_classes, search_space)

        return cls(input_shape=input_shape,
                   num_classes=num_classes,
                   layers=layers,
                   clf=clf,
                   history=History(None, 'random'),
                   num_epochs=num_epochs
                   )

    @classmethod
    def random_approach2(cls,
                         input_shape,
                         num_classes,
                         search_space=None):
        '''Returns a random CNN'''
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)

        depth_of_layers = np.random.randint(search_space.initialisation_range[0],
                                            search_space.initialisation_range[1]+1
                                            )

        # shape = np.asarray(input_shape, dtype='float')
        shape = input_shape
        layers = []
        while len(layers) < depth_of_layers:
            try:
                shape, layer = cls._create_random_layer(input_shape=shape,
                                                        search_space=search_space)
            except CannotReduceResolutionError:
                continue
            layers += [layer]

        num_epochs = np.random.randint(search_space.num_epoch_range[0],
                                       search_space.num_epoch_range[1]+1
                                       )

        # Add a random classifier
        clf = cls._create_random_classifier(shape, num_classes, search_space)

        return cls(input_shape=input_shape,
                   num_classes=num_classes,
                   layers=layers,
                   clf=clf,
                   history=History(None, 'random'),
                   num_epochs=num_epochs
                   )

    @classmethod
    def random(cls,
               input_shape,
               num_classes,
               search_space=None):
        '''Returns a random CNN'''
        search_space = search_space or cls.SearchSpace()
        assert isinstance(search_space, cls.SearchSpace)

        depth_of_layers = np.random.randint(search_space.initialisation_range[0],
                                            search_space.initialisation_range[1]+1
                                            )

        # shape = np.asarray(input_shape, dtype='float')
        shape = input_shape
        layers = []
        while len(layers) < depth_of_layers:
            try:
                shape, layer = cls._create_random_layer(input_shape=shape,
                                                        search_space=search_space)
            except CannotReduceResolutionError:
                break
            layers += [layer]

        num_epochs = np.random.randint(search_space.num_epoch_range[0],
                                       search_space.num_epoch_range[1]+1
                                       )

        # Add a random classifier
        clf = cls._create_random_classifier(shape, num_classes, search_space)

        return cls(input_shape=input_shape,
                   num_classes=num_classes,
                   layers=layers,
                   clf=clf,
                   history=History(None, 'random'),
                   num_epochs=num_epochs
                   )

    @staticmethod
    def _create_random_layer(input_shape, search_space, pooling_prob=None):
        assert isinstance(search_space, CNN.SearchSpace)

        if pooling_prob and PoolingLayer in search_space.layers:
            # pylint: disable=invalid-name
            Layer = PoolingLayer
            if np.random.rand() > pooling_prob and len(search_space.layers) > 1:
                while Layer == PoolingLayer:
                    Layer = np.random.choice(search_space.layers)
        else:
            Layer = np.random.choice(search_space.layers)

        # initialise that layer class
        layer = Layer.random(
            input_shape, search_space.dynamic_child_search_spaces[Layer])
        assert isinstance(layer, (DynamicLayer, DynamicClassifier))
        shape = layer.output_shape

        return shape, layer

    @staticmethod
    def _create_random_classifier(input_shape, num_classes, search_space):
        assert isinstance(search_space, CNN.SearchSpace)

        Clf = np.random.choice(  # pylint: disable=invalid-name
            search_space.clfs
        )
        layer = Clf.random(
            input_shape, search_space.dynamic_child_search_spaces[Clf])
        assert isinstance(layer, DynamicClassifier)

        layer.change_num_classes(num_classes)

        return layer

    def __init__(self,
                 input_shape,
                 num_classes,
                 layers,
                 clf,
                 num_epochs,
                 history: History
                 ):

        super(CNN, self).__init__()

        assert layers

        assert isinstance(history, History)
        assert isinstance(layers, list)
        assert isinstance(layers[0], DynamicLayer)
        assert isinstance(
            clf, DynamicClassifier), 'The last layer needs to be a classifier (i.E. SMMLP[])'

        self._input_shape = input_shape
        self._num_classes = num_classes
        self._layers = layers
        self._byte_size = None
        self._clf = clf
        self._num_classes = num_classes
        self.num_epochs = num_epochs
        self.history = history

        last_layer_shape = self._layers[-1].output_shape
        clf.change_input_shape(last_layer_shape)
        clf.change_num_classes(self._num_classes)

        self._sequential = torch.nn.Sequential(*(self._layers + [self._clf]))

    def clone(self):
        '''Returns an untrained clone of this CNN with the same topology'''
        return CNN(input_shape=self._input_shape,
                   num_classes=self._num_classes,
                   layers=[layer.clone() for layer in self._layers],
                   clf=self._clf.clone(),
                   history=History(
                       parent1=self,
                       reason='clone',
                   ),
                   num_epochs=self.num_epochs,
                   )

    def clone_with_weights(self):
        '''Returns a trained clone of this CNN'''
        clone = self.clone()
        err1, err2 = clone.load_state_dict(self.state_dict())
        assert not err1
        assert not err2

        return clone

    @property
    def num_classes(self):
        '''The output shape of this CNN'''
        return self._num_classes

    def forward(self, x):
        assert np.all(x.shape[1:] == self._input_shape)

        out = self._sequential(x)
        assert np.all(out.shape[1:][0] == self.num_classes)

        return out

    @property
    def size(self):
        '''Returns the size of this CNN in bytes

        Returns:
            int -- Size in bytes
        '''

        if not self._byte_size:
            self._byte_size = int(sum([np.product(v.size()) * v.element_size()
                                       for v in self.state_dict().values()]))

        return self._byte_size

    @property
    def topology(self):
        '''The string representation of this topology'''
        return '\n'.join([layer.topology for layer in self._layers] + [self._clf.topology, str(self.num_epochs)])

    def mutate(self, search_space=None):
        '''Mutates using the stochastic function from the search space'''
        search_space = search_space if search_space else CNN.SearchSpace()
        assert isinstance(search_space, CNN.SearchSpace)

        try:
            return search_space.mutate(self)
        except CannotReduceResolutionError:
            # If we try to add a pooling layer but it cannot pool anymore, restart mutation
            return self.mutate()

    # pylint: disable=invalid-name
    def insert_layer(self, search_space, Layer, index=None):
        '''Inserts a layer to a random position'''
        assert issubclass(Layer, DynamicLayer)
        assert isinstance(search_space, AbstractSearchSpace)

        def create_new_layer(shape):
            return Layer.random(shape, search_space=search_space)

        new_layers = self.insert_to_layers(
            layers=self._layers,
            input_shape=self._input_shape,
            create_new_layer=create_new_layer,
            index=index
        )

        return CNN(self._input_shape,
                   self._num_classes,
                   new_layers,
                   self._clf.clone(),
                   self.num_epochs,
                   history=History(
                       parent1=self,
                       reason='insert_layer_' + Layer.__name__
                   ),
                   )

    def remove_layer(self, _):
        '''Removes a layer at a random position'''
        if len(self._layers) < 2:
            # prevent removing the last layer
            return self.clone()

        return CNN(self._input_shape,
                   self._num_classes,
                   self.remove_from_layers(self._layers, self._input_shape),
                   self._clf.clone(),
                   self.num_epochs,
                   history=History(
                       parent1=self,
                       reason='remove_layer',
                   ))

    def change_layer(self, search_space):
        '''Mutates exactly one of the encapsulated layers'''
        search_space = search_space or self.SearchSpace()
        assert isinstance(search_space, CNN.SearchSpace)

        def mutator(layer, _):
            for Layer in search_space.dynamic_child_search_spaces.keys():
                if isinstance(layer, Layer):
                    return layer.mutate(search_space.dynamic_child_search_spaces[Layer])

            raise NotImplementedError(
                'Class %d not found in child search space' % layer.__class__)

        new_layers = self.mutate_in_layers(
            self._layers, self._input_shape, mutator)

        return CNN(self._input_shape,
                   self._num_classes,
                   new_layers,
                   self._clf.clone(),
                   self.num_epochs,
                   history=History(
                       parent1=self,
                       reason='change_layer',
                   ))

    def mutate_classifier(self, search_space):
        '''Mutates the classifier'''
        search_space = search_space or self.SearchSpace()
        assert isinstance(search_space, CNN.SearchSpace)

        search_space = search_space.dynamic_child_search_spaces[
            type(self._clf)
        ]

        clf = self._clf.mutate(search_space)
        layers = [l.clone() for l in self._layers]

        return CNN(
            input_shape=self._input_shape,
            num_classes=self._num_classes,
            layers=layers,
            clf=clf,
            num_epochs=self.num_epochs,
            history=History(
                parent1=self,
                reason='mutate_clf',
            ))

    def mutate_training_epochs(self, search_space):
        '''Change training epoch'''
        search_space = search_space or self.SearchSpace()
        assert isinstance(search_space, CNN.SearchSpace)

        old_epoch = self.num_epochs
        num_epochs = max(0, search_space.num_epoch_mutation(self.num_epochs))
        diff = num_epochs - old_epoch

        return CNN(
            input_shape=self._input_shape,
            num_classes=self._num_classes,
            layers=[l.clone() for l in self._layers],
            clf=self._clf,
            num_epochs=num_epochs,
            history=History(
                parent1=self,
                reason='mutate_epoch: %d' % diff,
            ))

    def one_point_crossover(self, other):
        """Cross-over two CNNs and return two individuals.
        Cross-over location does not normally land on the extremes (0 or max)
        Except for individuals with exactly one layer of course
        The classifier is picked at random
        """
        # pylint: disable=protected-access

        assert isinstance(self, CNN)
        assert isinstance(other, CNN)
        assert self._input_shape == other._input_shape
        assert self._num_classes == other._num_classes

        (layers1, layers2), (point1, point2) = self.one_point_crossover_layers(
            self._input_shape, self._layers, other._layers)

        # one-point crossover will return the first half of the first individual first
        # so we can chose the matching classifiers
        clf1 = self._clf.clone()
        clf2 = other._clf.clone()

        history1 = History(
            parent1=self,
            parent2=other,
            cross_over_point1=point1,
            cross_over_point2=point2,
            reason='cross_over_1',
        )

        history2 = History(
            parent1=other,
            parent2=self,
            cross_over_point1=point2,
            cross_over_point2=point1,
            reason='cross_over_2',
        )

        # mix the num_epochs randomly, do not bind it to the SMMLP
        if np.random.rand() < .5:
            num_epochs1 = self.num_epochs
            num_epochs2 = other.num_epochs
        else:
            num_epochs1 = other.num_epochs
            num_epochs2 = self.num_epochs

        return (CNN(self._input_shape, self._num_classes, layers1, clf2, num_epochs1, history=history1),
                CNN(other._input_shape, other._num_classes, layers2, clf1, num_epochs2, history=history2))
