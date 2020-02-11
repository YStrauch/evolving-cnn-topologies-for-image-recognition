# pylint: disable=protected-access, missing-docstring
import unittest
import helpers
import cga


class Kernel(helpers.Test):
    def test_serialisation(self):
        comp = cga.Kernel.random(
            [3, 64, 64],
            cga.Kernel.SearchSpace([1], [(1, 1)], [(1, 1)], ['V']))
        string_representation = comp.topology
        self.assertIsInstance(string_representation, str)

        comp = cga.Kernel.from_string(string_representation, [3, 64, 64])
        self.assertEqual(comp.topology, string_representation)

    def test_mutation(self):
        search_space1 = cga.Kernel.SearchSpace(
            depths=[1],
            resolutions=[(2, 3)],
            strides=[(4, 4)],
            padding_types=['V']
        )
        search_space2 = cga.Kernel.SearchSpace(
            depths=[100],
            resolutions=[(200, 300)],
            strides=[(400, 400)],
            padding_types=['S']
        )

        filter1 = cga.Kernel.random([64, 32, 32], search_space1)
        filter2 = filter1.mutate(search_space2)

        self.assertNotEqual(filter1, filter2)
        self.assertEqual(filter1.depth, 1)
        self.assertEqual(filter2.depth, 100)
        self.assertNotEqual(filter1.same_padding, filter2.same_padding)

    def test_random(self):
        search_space = cga.Kernel.SearchSpace(
            depths=[1],
            resolutions=[(2, 3)],
            strides=[(4, 10)],
            padding_types=['V']
        )

        filter1 = cga.Kernel.random([3, 64, 64], search_space)
        self.assertEqual(filter1.topology, '001(2x3 4x10 V)')
        self.assertArrayEqual(filter1.resolution, [2, 3])
        self.assertArrayEqual(filter1.stride, [4, 10])
        self.assertEqual(filter1.depth, 1)
        self.assertEqual(filter1.same_padding, False)

        search_space = cga.Kernel.SearchSpace(
            depths=[100],
            resolutions=[(5, 1)],
            strides=[(2, 2)],
            padding_types=['S']
        )
        filter2 = cga.Kernel.random([3, 64, 64], search_space)
        self.assertEqual(filter2.topology, '100(5x1 2x2 S)')
        self.assertArrayEqual(filter2.resolution, [5, 1])
        self.assertArrayEqual(filter2.stride, [2, 2])
        self.assertEqual(filter2.depth, 100)
        self.assertEqual(filter2.same_padding, True)

    def test_from_string_with_depth(self):
        topology = '064(3x3 1x1 S)'
        filt = cga.Kernel.from_string(topology, (64, 64))
        self.assertEqual(filt.topology, topology)
        self.assertEqual(filt.depth, 64)
        self.assertArrayEqual(filt.stride, (1, 1))
        self.assertArrayEqual(filt.resolution, (3, 3))

    def test_from_string_without_depth(self):
        topology = '3x3 1x1 S'
        filt = cga.Kernel.from_string(topology, (64, 64))
        self.assertEqual(filt.topology, topology)
        self.assertArrayEqual(filt.stride, (1, 1))
        self.assertArrayEqual(filt.resolution, (3, 3))
        self.assertEqual(filt.depth, None)

    def test_clone(self):
        filter1 = cga.Kernel.random([3, 64, 64],
                                    cga.Kernel.SearchSpace([1], [(1, 1)], [(1, 1)], ['V']))

        self.assertNotEqual(filter1.clone(), filter1)


if __name__ == '__main__':
    unittest.main()
