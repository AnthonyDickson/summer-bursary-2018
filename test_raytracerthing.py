import unittest

import numpy as np

from raytracerthing import RayTracerThing


class TestRaytracerThing(unittest.TestCase):

    def test_output(self):
        image = np.array([[0, 1],
                          [2, 3]])

        expected = np.array([[6, 6],
                             [6, 6]])

        the_thing = RayTracerThing(input_shape=(2, 2),
                                   output_shape=(2, 2))

        actual = the_thing.forward(image)

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected an output of %s, instead got %s."
                        % (expected, actual))


if __name__ == '__main__':
    unittest.main()
