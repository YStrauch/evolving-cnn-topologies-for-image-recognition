'''Base classes that are used by other classes'''
from abc import ABC, abstractmethod
import numpy as np

import torch.nn as nn


class CannotReduceResolutionError(Exception):
    '''Signalises that an algorithm tried to add a new pooling layer but it is impossible'''


class InvalidTopologyError(Exception):
    '''Signalises that the topology is syntactically incorrect'''


class GeneticComponent(ABC):
    '''A Genetic Component can mutate and is serialisable'''

    @classmethod
    @abstractmethod
    def from_string(cls, string, input_shape):
        '''Creates a layer from a string with an optional context'''

    @classmethod
    @abstractmethod
    def random(cls, input_shape, search_space=None):
        '''Creates a layer from random'''

    @abstractmethod
    def clone(self):
        '''Returns an untrained clone of this layer with the same topology'''

    @abstractmethod
    def mutate(self, search_space=None):
        '''Mutates this layer'''

    @property
    @abstractmethod
    def topology(self):
        '''Returns a string representation that can be put into from_string()'''


class AbstractSearchSpace(ABC):
    '''Defines how a dynamic module draws random configurations'''

    def reset(self):
        '''Resets this search space'''
        self.__init__()


class StochasticFunction():
    '''Holds a set of probabilistic callbacks, call the function for a random callback'''

    def __init__(self, d):
        assert isinstance(d, dict)
        probs = np.array(list(d.values()))

        probsum = np.sum(probs)
        if probsum != 1:
            # normalise probabilities so they add up to 1
            probs = probs / probsum

        self._probs = probs
        self._funcs = list(d.keys())

    def __call__(self, *args, **kwargs):
        probs = np.cumsum(np.asarray(self._probs, dtype=float))
        func = self._funcs[list(np.random.random() < probs).index(True)]
        return func(*args, **kwargs)


class LayerContainer(nn.Module):
    '''Wraps and manages other layers, i.E. CNN, Skip Layer'''
    @staticmethod
    def mutate_in_layers(layers, input_shape, manipulator, index=None):
        """
        Helper function to mutate a layer easily.
        Will clone everything to ensure immutables.
        Will propagate output shapes to all predecessors.
        """
        index = index if index is not None else (
            np.random.randint(0, len(layers)) if layers else 0)

        before = layers[:index]
        # index might be bigger than list size if we want to insert
        selected = layers[index].clone() if index < len(layers) else None
        after = layers[index+1:]

        shape = before[-1].output_shape if before else input_shape
        manipulated = manipulator(selected, shape)
        # manipulated may be one or more objects, normalise that
        try:
            manipulated = [*manipulated]
        except TypeError:
            manipulated = [manipulated]

        # now glue them back together
        # everything before doesn't need to change shape
        parsed = [l.clone() for l in before]

        # then add the manipulated layer(s)
        if manipulated:
            parsed = parsed + manipulated

        # we need to pull the shape again because the mutation might have changed it
        shape = parsed[-1].output_shape if parsed else input_shape

        for layer in after:
            layer = layer.clone()
            layer.change_input_shape(shape)
            shape = layer.output_shape
            parsed.append(layer)

        return parsed

    @classmethod
    def remove_from_layers(cls, layers, input_shape, index=None):
        """
        Allows to remove a layer easily.
        Will clone everything to ensure immutables.
        Will propagate output shapes to all predecessors.
        """

        def manipulator(_, __):
            return []  # drops the selected layer

        return cls.mutate_in_layers(layers, input_shape, manipulator, index)

    @classmethod
    def insert_to_layers(cls, layers, input_shape, create_new_layer, index=None):
        """
        Allows to add a layer easily.
        Will clone everything to ensure immutables.
        Will propagate output shapes to all predecessors.

        Raises: CannotReduceResolutionError
        """
        # note that this time our random is bound to len+1 so we can append to the end
        index = index if index is not None else (
            np.random.randint(0, len(layers) + 1) if layers else 0
        )

        def manipulator(following_layer, input_shape):
            new_layer = create_new_layer(input_shape)

            # when prepending/inserting, the following layer must be notified
            # following layer may be None if we are appending
            if following_layer:  # necessary because our layer might be the last one
                following_layer.change_input_shape(new_layer.output_shape)
                return [new_layer, following_layer]

            return [new_layer]

        return cls.mutate_in_layers(layers, input_shape, manipulator, index)

    def one_point_crossover_layers(self, input_shape, layers1, layers2, point1=None, point2=None):
        """
        One-point cross-over between two layer stack
        Will return two layer stacks again
        """
        def random_index(layers):
            # if there is more than one layer, set the min to 1
            # this prevents cross-over to simply remove a bit

            min_index = 1 if len(layers) > 1 else 0
            return np.random.randint(min_index, len(layers))

        def propagate_shapes(crossover_point, input_shape, layers):
            '''Clones all layers and propagates shapes from the crossover point downwards'''
            if crossover_point == 0:
                shape = input_shape
            else:
                shape = layers[crossover_point-1].output_shape

            processed_layers = [layer.clone()
                                for layer in layers[:crossover_point]]

            for layer in layers[crossover_point:]:
                layer = layer.clone()
                layer.change_input_shape(shape)
                shape = layer.output_shape
                processed_layers.append(layer)

            return processed_layers

        point1 = random_index(layers1) if point1 is None else point1
        point2 = random_index(layers2) if point2 is None else point2

        # cross-over
        cross1 = layers1[:point1] + layers2[point2:]
        cross2 = layers2[:point2] + layers1[point1:]

        # shapes
        cross1 = propagate_shapes(point1, input_shape, cross1)
        cross2 = propagate_shapes(point2, input_shape, cross2)

        return (cross1, cross2), (point1, point2)

    def forward(self, *x):
        raise NotImplementedError


class DynamicLayer(nn.Module, GeneticComponent):
    '''A dynamic layer within a CNN'''

    @staticmethod
    @abstractmethod
    def get_layer_type():
        '''Returns the unique layer type'''

    @property
    @abstractmethod
    def output_shape(self):
        '''The output shape of this layer'''

    @abstractmethod
    def change_input_shape(self, new_input_shape):
        '''Notifies the layer that its input shape has changed'''


class DynamicClassifier(nn.Module, GeneticComponent):
    '''A dynamic classifier at the end of a CNN'''

    @staticmethod
    @abstractmethod
    def get_layer_type():
        '''Returns the unique layer type'''

    @property
    @abstractmethod
    def num_classes(self):
        '''How many classes this classifier outputs'''

    @abstractmethod
    def change_input_shape(self, new_input_shape):
        '''Notifies the classifier that its input shape has changed'''

    @abstractmethod
    def change_num_classes(self, new_num_classes):
        '''Notifies the classifier that its number of classes changed'''
