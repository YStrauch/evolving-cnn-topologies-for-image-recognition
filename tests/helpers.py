# pylint: disable=protected-access, missing-docstring

import logging
import sys
import os
import unittest
from unittest.util import safe_repr

import numpy as np
import torch
sys.path.insert(0, os.path.curdir)


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG
)


def serialisation_works(component, **args):
    '''Checks that a component can serialise and unserialise'''
    comp = component.random(**args)
    string_representation = comp.topology
    if not isinstance(string_representation, str):
        return False

    comp = component.from_string(string_representation, **args)
    return comp.topology == string_representation

    # assert that clone emits a new individual
    # return comp != comp.clone()


class Test(unittest.TestCase):
    # pylint: disable=invalid-name
    def assertArrayEqual(self, first, second, msg=None):
        '''Fail if a is not a numpy array or if it is not equal to b'''

        if not isinstance(first, np.ndarray):
            std_msg = '%s is %s, not numpy array' % (
                safe_repr(first), type(first))
            self.fail(self._formatMessage(msg, std_msg))

        if not np.all(first == second):
            std_msg = '%s is not equal to %s' % (
                safe_repr(first), safe_repr(second)
            )
            self.fail(self._formatMessage(msg, std_msg))

    # pylint: disable=invalid-name
    def assertTensorShape(self, tensor, shape, msg=None):
        '''Fail if a is not a torch tensor or if its shape is wrong'''

        if not isinstance(tensor, torch.Tensor):
            std_msg = '%s is %s, not tensor' % (
                safe_repr(tensor), type(tensor))
            self.fail(self._formatMessage(msg, std_msg))

        if not np.all(tensor.shape == torch.Size(shape)):
            std_msg = 'Shape %s is not equal to %s' % (
                tensor.shape, torch.Size(shape)
            )
            self.fail(self._formatMessage(msg, std_msg))
