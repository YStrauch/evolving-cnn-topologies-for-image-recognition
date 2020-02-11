'''Filter / Kernel that is used by the layers'''
import re

import numpy as np

from cga_src.base import GeneticComponent, CannotReduceResolutionError, AbstractSearchSpace


class Kernel(GeneticComponent):
    '''A filter / Kernel of a CNN'''

    class SearchSpace(AbstractSearchSpace):
        '''Defines the parameters and their space'''

        def __init__(self, depths, resolutions, strides, padding_types):
            self.depths = depths
            self.resolutions = resolutions
            self.strides = strides
            self.padding_types = padding_types

    @classmethod
    def from_string(cls, string, input_shape):
        '''Creates a filter / kernel from a string representation'''

        if '(' in string:
            depth, kernel_x, kernel_y, stride_x, stride_y, padding_type = re.fullmatch(
                r'^(\d+)\((\d+)x(\d+) (\d+)x(\d+) ([SV])\)$', string
            ).group(*range(1, 7))
            depth = int(depth)
        else:
            # no depth
            kernel_x, kernel_y, stride_x, stride_y, padding_type = re.fullmatch(
                r'^(\d+)x(\d+) (\d+)x(\d+) ([SV])$', string
            ).group(*range(1, 6))
            depth = None

        kernel = int(kernel_x), int(kernel_y)
        stride = int(stride_x), int(stride_y)

        return cls(input_shape, depth, kernel, stride, padding_type == 'S')

    @classmethod
    def random(cls, input_shape, search_space=None):
        '''Creates a random filter'''
        assert isinstance(search_space, cls.SearchSpace)

        depth = np.random.choice(search_space.depths)
        resolution = search_space.resolutions[np.random.randint(
            0, len(search_space.resolutions))]
        stride = search_space.strides[np.random.randint(
            0, len(search_space.strides))]
        same_padding = np.random.choice(search_space.padding_types) == 'S'

        return cls(input_shape, depth, resolution, stride, same_padding)

    def __init__(self, input_shape, depth, resolution, stride, same_padding):
        self._input_shape = input_shape
        self._depth = depth
        self._resolution = np.asarray(resolution)
        self._stride = np.asarray(stride)
        self._same_padding = np.asarray(same_padding)

    @property
    def input_shape(self):
        '''The input_shape of filters/kernels'''
        return self._input_shape

    @property
    def depth(self):
        '''The depth of filters/kernels'''
        return self._depth

    @property
    def resolution(self):
        '''The resolution of filters/kernels'''
        return self._resolution

    @property
    def stride(self):
        '''The stride of filters/kernels'''
        return self._stride

    @property
    def same_padding(self):
        '''If the filter/kernel uses same_padding'''
        return self._same_padding

    def clone(self):
        '''Returns a component with the same topology but untrained weights'''
        return Kernel(
            self.input_shape,
            self.depth,
            self.resolution,
            self.stride,
            self.same_padding
        )

    def calc_output_shape_and_padding(self):
        '''Calculates the output shape'''
        assert np.all(self.input_shape == np.asarray(
            self.input_shape, dtype=int))
        input_shape = np.asarray(self.input_shape, dtype=float)

        padding = self._calc_same_padding(
            input_shape) if self._same_padding else np.asarray((0, 0))
        input_resolution = input_shape[1:]

        output_resolution = input_resolution - self.resolution + 2 * padding
        output_resolution = output_resolution / self.stride + [1, 1]
        # filters=None used for pooling => reuse input depth
        depth = self.depth or input_shape[0]
        output_shape = np.insert(output_resolution, 0, depth)

        if np.any(output_shape != np.round(np.array(output_shape))) or np.any(output_shape <= 0):
            raise CannotReduceResolutionError

        output_shape = output_shape.astype(int)

        return output_shape, padding

    def _calc_same_padding(self, input_shape):
        resolution = input_shape[1:]

        padding = (self.stride*(resolution-1) + self.resolution - resolution)/2
        padding_int = padding.astype(int)

        if not np.all(padding == padding_int):
            # TODO: asymmetric padding
            raise NotImplementedError(
                'Asymmetric SAME padding not yet implemented')

        return padding_int

    def change_input_shape(self, input_shape):
        self._input_shape = input_shape

    def mutate(self, search_space=None):
        '''Mutates this filter given the search space'''
        return self.random(self._input_shape, search_space)

    @property
    def topology(self):
        '''String representation of this topology'''

        if self._depth:
            return '%.3d(%dx%d %dx%d %s)' % (
                self._depth,
                self._resolution[0], self._resolution[1],
                self._stride[0], self._stride[1],
                'S' if self._same_padding else 'V'
            )

        return '%dx%d %dx%d %s' % (
            self._resolution[0], self._resolution[1],
            self._stride[0], self._stride[1],
            'S' if self._same_padding else 'V'
        )
