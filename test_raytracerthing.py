import unittest

import numpy as np

from raytracerthing import RayTracerThing, PixelGrid
from raytracing import Vec3f, Ray3D


class TestPixelGrid(unittest.TestCase):

    def test_init(self):
        shape = (2, 2)
        values = [[0, 1],
                  [2, 3]]

        pg = PixelGrid(*shape, pixel_values=values)

        self.assertEqual(shape, pg.shape)
        self.assertTrue(np.array_equal(values, pg.pixel_values))
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
            actual = pg.to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_2x2grid_coords_conversion_right_edge(self):
        shape = (2, 2)

        pg = PixelGrid(*shape)

        # points are defined at the top right corner of each pixel.
        points = [
            (0, 1),
            (2, 1),
            (0, -1),
            (2, -1)
        ]

        # Points that lie on overlapping borders cause the resulting column to be that of the pixel to the right.
        expected_grid_coords = [
            (0, 1),
            (0, 1),
            (1, 1),
            (1, 1)
        ]

        for (x, y), expected in zip(points, expected_grid_coords):
            actual = pg.to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_grid_coords_4way_intersection(self):
        shape = (2, 2)

        pg = PixelGrid(*shape)

        x, y = (0, 0)  # right in the middle of the grid where all four pixels meet.
        # when the point intersects where pixels are overlapping, the row/column is resolved to that of the pixel
        # below/to the right.
        expected = (1, 1)
        actual = pg.to_grid_coords(x, y)

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
            actual = pg.to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_grid_coords_wider_pixels(self):
        shape = (4, 4)
        pixel_size = 2

        pg = PixelGrid(*shape, pixel_size=pixel_size)

        top_left = (pg.top_left.x, pg.top_left.y)
        points = []
        expected_grid_coords = []

        for row in range(shape[0]):
            for col in range(shape[1]):
                points.append((top_left[0] + col * pixel_size, top_left[1] - row * pixel_size))
                expected_grid_coords.append((row, col))

        for (x, y), expected in zip(points, expected_grid_coords):
            actual = pg.to_grid_coords(x, y)

            self.assertEqual(expected, actual,
                             "Expected the point (%d, %d) to translate to the "
                             "grid coords (row, col) [%s], instead got %s."
                             % (x, y, expected, actual))

    def test_pixel_extents(self):
        shape = (2, 2)

        pg = PixelGrid(*shape)

        expected_extents = [
            [(Vec3f([-1, 1, 0]), Vec3f([0, 0, 0])), (Vec3f([0, 1, 0]), Vec3f([1, 0, 0]))],
            [(Vec3f([-1, 0, 0]), Vec3f([0, -1, 0])), (Vec3f([0, 0, 0]), Vec3f([1, -1, 0]))],
        ]

        for row in range(pg.n_rows):
            for col in range(pg.n_cols):
                expected_min_extent, expected_max_extent = expected_extents[row][col]
                actual_min_extent, actual_max_extent = pg.pixel_extents[row][col]

                min_extents_matching = np.allclose(expected_min_extent, actual_min_extent)
                max_extents_matching = np.allclose(expected_max_extent, actual_max_extent)

                self.assertTrue(min_extents_matching and max_extents_matching,
                                'Expected pixel at [%d, %d] (row, col) to '
                                'have its extents at %s and %s, instead got '
                                '%s and %s.' % (row, col,
                                                expected_min_extent,
                                                expected_max_extent,
                                                actual_min_extent,
                                                actual_max_extent))

    def test_pixel_centers(self):
        shape = (2, 2)

        pg = PixelGrid(*shape)

        expected_centers = [
            [Vec3f([-0.5, 0.5, 0]), Vec3f([0.5, 0.5, 0])],
            [Vec3f([-0.5, -0.5, 0]), Vec3f([0.5, -0.5, 0])]
        ]

        for row in range(pg.n_rows):
            for col in range(pg.n_cols):
                expected = expected_centers[row][col]
                actual = pg.pixel_centers[row][col]

                self.assertTrue(np.allclose(expected, actual),
                                'Expected pixel at [%d, %d] (row, col) to '
                                'have its center point at %s, instead got '
                                '%s.' % (row, col, expected, actual))

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


class TestRaytracerThing(unittest.TestCase):

    def test_output_no_hidden_layers(self):
        image = np.array([[0, 1],
                          [2, 3]])

        expected = np.array([[6, 6],
                             [6, 6]])

        the_thing = RayTracerThing(input_shape=(2, 2),
                                   output_shape=(2, 2),
                                   n_layers=0)
        the_thing.enable_full_transparency()

        actual = the_thing.forward(image)

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected an output of %s, instead got %s."
                        % (expected, actual))

    def test_output_full_opacity(self):
        image = np.array([[0, 1],
                          [2, 3]])

        expected = np.array([[0, 0],
                             [0, 0]])

        the_thing = RayTracerThing(input_shape=(2, 2),
                                   output_shape=(2, 2),
                                   n_layers=1)
        the_thing.enable_full_opacity()

        actual = the_thing.forward(image)

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected an output of %s, instead got %s."
                        % (expected, actual))

    def test_output_one_hidden_layer(self):
        image = np.array([[0, 1],
                          [2, 3]])

        expected = image

        the_thing = RayTracerThing(input_shape=(2, 2),
                                   output_shape=(2, 2),
                                   n_layers=1)
        the_thing.enable_full_transparency()

        actual = the_thing.forward(image)

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected an output of %s, instead got %s."
                        % (expected, actual))


if __name__ == '__main__':
    unittest.main()
