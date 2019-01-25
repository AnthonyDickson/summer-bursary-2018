import unittest

import numpy as np

from raytracerthing import RayTracerThing, PixelGrid
from raytracing import Vec3f, Ray3D


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


class TestPixelGrid(unittest.TestCase):

    def test_init(self):
        shape = (2, 2)
        values = [[0, 1],
                  [2, 3]]

        pg = PixelGrid(*shape, pixel_values=values)

        self.assertEqual(shape, pg.shape)
        self.assertTrue(np.array_equal(values, pg.pixels))
        self.assertTrue(np.array_equal(Vec3f.zero(), pg.origin))
        self.assertTrue(np.array_equal(Vec3f([-1, -1, 0]), pg.bottom_left))
        self.assertTrue(np.array_equal(Vec3f([1, 1, 0]), pg.top_right))

    def test_2x2grid_coords_conversion(self):
        shape = (2, 2)

        pg = PixelGrid(*shape)

        points = [
            (-1, 1),
            (1, 1),
            (-1, -1),
            (1, -1)
        ]

        expected_grid_coords = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1)
        ]

        for (x, y), expected in zip(points, expected_grid_coords):
            actual = pg.xy_to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_4x4grid_coords_conversion(self):
        shape = (4, 4)

        pg = PixelGrid(*shape)

        top_left = (-0.5 * shape[0], 0.5 * shape[1])
        points = []
        expected_grid_coords = []

        for row in range(shape[0]):
            for col in range(shape[1]):
                points.append((top_left[0] + col, top_left[1] - row))
                expected_grid_coords.append((row, col))

        for (x, y), expected in zip(points, expected_grid_coords):
            actual = pg.xy_to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_hit_value(self):
        shape = (2, 2)
        values = [[0, 1],
                  [2, 3]]

        pg = PixelGrid(*shape, pixel_values=values)

        corners = [[pg.top_left, pg.top_right],
                   [pg.bottom_left, pg.bottom_right]]

        for row in range(shape[0]):
            for col in range(shape[1]):
                ray = Ray3D(origin=corners[row][col], direction=Vec3f.forward())

                expected = values[row][col]
                actual = pg.hit_value(ray)

                self.assertEqual(expected, actual,
                                 "Expected the pixel at grid position "
                                 "[%d, %d] (row, col) to be %d, "
                                 "instead got %d with a ray originating at %s "
                                 "and pointing in the direction %s.)"
                                 % (row, col, expected, actual, ray.origin,
                                    ray.direction))


if __name__ == '__main__':
    unittest.main()
