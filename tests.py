import unittest

import numpy as np

from raytracerthing import RayTracerThing, Vec3D, Ray


class TestVec3D(unittest.TestCase):

    def test_init(self):
        expected = [0, 0, 0]
        actual = Vec3D.zero()

        self.assertTrue(np.array_equal(expected, actual))

    def test_init_raises_error(self):
        with self.assertRaises(AssertionError):
            Vec3D([1])
            Vec3D([1, 2, 3, 4])

    def test_xyz_accessors(self):
        v = Vec3D([1, 2, 3])

        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def test_normalise_zero(self):
        v = Vec3D.zero()

        expected = [0, 0, 0]
        actual = v.normalise()

        self.assertTrue(np.array_equal(expected, actual))

    def test_normalise_simple(self):
        v = Vec3D([0, 0, 1])

        expected = [0, 0, 1]
        actual = v.normalise()

        self.assertTrue(np.array_equal(expected, actual))

    def test_normalise(self):
        v = Vec3D([1, 2, 3])

        expected = [0.26726124, 0.53452248, 0.80178373]
        actual = v.normalise()

        self.assertTrue(np.allclose(expected, actual))


class TestRay(unittest.TestCase):

    def test_ray_at_origin(self):
        ray = Ray()

        expected = np.array([0, 0, 0])
        actual = ray.get_point(t=0)

        self.assertTrue(np.array_equal(expected, actual))

    def test_ray_position(self):
        ray = Ray(direction=Vec3D([0, 0, 1]))

        expected = np.array([0, 0, 1])
        actual = ray.get_point(t=1)

        self.assertTrue(np.array_equal(expected, actual))

        expected = np.array([0, 0, 2.5])
        actual = ray.get_point(t=2.5)

        self.assertTrue(np.array_equal(expected, actual))


class TestRaytracerThing(unittest.TestCase):

    def test_output(self):
        image = np.array([[0, 1],
                          [2, 3]])

        expected = np.array([[6, 6],
                             [6, 6]])

        the_thing = RayTracerThing(input_shape=(2, 2),
                                   output_shape=(2, 2))

        actual = the_thing.forward(image)

        self.assertTrue(np.array_equal(expected, actual))


if __name__ == '__main__':
    unittest.main()
