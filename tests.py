import unittest

import numpy as np

from raytracerthing import RayTracerThing, Vec3f, Ray, Plane3D


class TestVec3D(unittest.TestCase):

    def test_init(self):
        expected = [0, 0, 0]
        actual = Vec3f.zero()

        self.assertTrue(np.array_equal(expected, actual))

    def test_init_raises_error(self):
        with self.assertRaises(AssertionError):
            Vec3f([1])
            Vec3f([1, 2, 3, 4])

    def test_xyz_accessors(self):
        v = Vec3f([1, 2, 3])

        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def test_normalise_zero(self):
        v = Vec3f.zero()

        expected = [0, 0, 0]
        actual = v.normalise()

        self.assertTrue(np.array_equal(expected, actual))

    def test_normalise_simple(self):
        v = Vec3f([0, 0, 1])

        expected = [0, 0, 1]
        actual = v.normalise()

        self.assertTrue(np.array_equal(expected, actual))

    def test_normalise(self):
        v = Vec3f([1, 2, 3])

        expected = [0.26726124, 0.53452248, 0.80178373]
        actual = v.normalise()

        self.assertTrue(np.allclose(expected, actual))


class TestPlane3D(unittest.TestCase):

    def test_init(self):
        plane = Plane3D()

        expected = 0
        actual = plane.d

        self.assertEqual(expected, actual)

        plane = Plane3D(origin=Vec3f([1, 0, 1]),
                        norm=Vec3f([1, 0, 0]))

        expected = 1
        actual = plane.d

        self.assertEqual(expected, actual)

    def test_init_raises_error(self):
        with self.assertRaises(AssertionError):
            Plane3D(origin=Vec3f.zero(),
                    norm=Vec3f.zero())

    def test_point_intersection(self):
        plane = Plane3D()  # any point with z=0 should intersect.
        point = Vec3f([1, 3, 0])

        self.assertTrue(plane.point_intersects(point))


class TestRay(unittest.TestCase):

    def test_ray_at_origin(self):
        ray = Ray()

        expected = np.array([0, 0, 0])
        actual = ray.get_point(t=0)

        self.assertTrue(np.array_equal(expected, actual))

    def test_ray_position(self):
        ray = Ray(direction=Vec3f([0, 0, 1]))

        expected = np.array([0, 0, 1])
        actual = ray.get_point(t=1)

        self.assertTrue(np.array_equal(expected, actual))

        expected = np.array([0, 0, 2.5])
        actual = ray.get_point(t=2.5)

        self.assertTrue(np.array_equal(expected, actual))

    def test_ray_plane_single_intersection(self):
        ray = Ray(origin=Vec3f([0, 0, -1]),
                  direction=Vec3f([0, 0, 1]))

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f([0, 0, 1]))

        self.assertTrue(ray.intersects_plane(plane))

        # This ray should be close to parallel, but still intersect.
        ray = Ray(origin=Vec3f([0, 0, -1]),
                  direction=Vec3f([10000000, 0, 1]))

        self.assertTrue(ray.intersects_plane(plane))

    def test_ray_plane_contained_intersection(self):
        ray = Ray(origin=Vec3f.zero(),
                  direction=Vec3f([1, 0, 0]))

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f([0, 0, 1]))

        self.assertTrue(ray.intersects_plane(plane))

    def test_ray_plane_no_intersection(self):
        ray = Ray(origin=Vec3f([0, 0, -1]),
                  direction=Vec3f([1, 0, 0]))

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f([0, 0, 1]))

        self.assertFalse(ray.intersects_plane(plane))


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
