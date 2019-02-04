import unittest

import numpy as np

from raytracerthing import RayTracerThing


class TestRaytracerThing(unittest.TestCase):

    def test_output_no_hidden_layers(self):
        image = np.array([[0.0, 1.0],
                          [2.0, 3.0]])

        expected = 6.0

        the_thing = RayTracerThing(input_shape=(2, 2), n_layers=0)
        the_thing.enable_full_transparency()

        actual = the_thing.predict(image).numpy()

        self.assertEqual(expected, actual,
                         "Expected an output of %s, instead got %s."
                         % (expected, actual))

    def test_output_full_opacity(self):
        image = np.array([[0.0, 1.0],
                          [2.0, 3.0]])

        expected = 0.0

        the_thing = RayTracerThing(input_shape=(2, 2), n_layers=1)
        the_thing.enable_full_opacity()

        actual = the_thing.predict(image).detach().numpy()

        self.assertEqual(expected, actual,
                         "Expected an output of %s, instead got %s."
                         % (expected, actual))

    def test_output_one_hidden_layer(self):
        image = np.array([[0.0, 1.0],
                          [2.0, 3.0]])

        expected = 6.0

        the_thing = RayTracerThing(input_shape=(2, 2), n_layers=1)
        the_thing.enable_full_transparency()

        actual = the_thing.predict(image).detach().numpy()

        self.assertEqual(expected, actual,
                         "Expected an output of %s, instead got %s."
                         % (expected, actual))


if __name__ == '__main__':
    unittest.main()
