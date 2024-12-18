import unittest

import numpy as np

from raytracing import Vec3f, Plane3D, Box3D, Ray3D


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

        expected = Vec3f.zero()
        actual = v.normalise()

        self.assertTrue(np.array_equal(expected, actual))

    def test_normalise_simple(self):
        v = Vec3f.forward()

        expected = Vec3f.forward()
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
                        norm=Vec3f.right())

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

        self.assertTrue(plane.contains(point))


class TestBox3D(unittest.TestCase):

    def test_init(self):
        box = Box3D()

        expected = Vec3f.zero()
        actual = box.centroid

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected box centroid to be at the point %s, but "
                        "instead it was found at the point %s"
                        % (expected, actual))

        expected = [1.0, 1.0, 1.0]
        actual = box.dimensions

        self.assertTrue(np.array_equal(expected, actual),
                        "Expected box dimensions to be %s, "
                        "but instead it was  %s"
                        % (expected, actual))

    def test_contains_point(self):
        box = 2 * Box3D()  # unit cube scaled by 2.

        point = Vec3f.zero()

        self.assertTrue(box.contains(point))

        point = Vec3f([1, 1, 1])

        self.assertTrue(box.contains(point))

        point = Vec3f([1, 1, 1.00000000000001])

        self.assertFalse(box.contains(point))


class TestRay(unittest.TestCase):

    def test_ray_at_origin(self):
        ray = Ray3D()

        expected = Vec3f.zero()
        actual = ray.get_point(t=0)

        self.assertTrue(np.array_equal(expected, actual))

    def test_ray_position(self):
        ray = Ray3D(direction=Vec3f.forward())

        expected = Vec3f.forward()
        actual = ray.get_point(t=1)

        self.assertTrue(np.array_equal(expected, actual))

        expected = np.array([0, 0, 2.5])
        actual = ray.get_point(t=2.5)

        self.assertTrue(np.array_equal(expected, actual))

    def test_ray_plane_single_intersection(self):
        ray = Ray3D(origin=Vec3f.backwards(),
                    direction=Vec3f.forward())

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f.forward())

        self.assertTrue(ray.intersects(plane))

        # This ray should be close to parallel, but still intersect.
        ray = Ray3D(origin=Vec3f.backwards(),
                    direction=Vec3f([10000000, 0, 1]))

        self.assertTrue(ray.intersects(plane))

    def test_ray_plane_contained_intersection(self):
        ray = Ray3D(origin=Vec3f.zero(),
                    direction=Vec3f.right())

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f.forward())

        self.assertTrue(ray.intersects(plane))

    def test_ray_plane_no_intersection(self):
        ray = Ray3D(origin=Vec3f.backwards(),
                    direction=Vec3f.right())

        plane = Plane3D(origin=Vec3f.zero(),
                        norm=Vec3f.forward())

        self.assertFalse(ray.intersects(plane))

        ray = Ray3D(origin=Vec3f.backwards(),
                    direction=Vec3f.left())

        self.assertFalse(ray.intersects(plane))

    def test_ray_box_intersection(self):
        hit_rays = [
            Ray3D(origin=Vec3f.right(), direction=Vec3f.left()),
            Ray3D(origin=Vec3f.up(), direction=Vec3f.down()),
            Ray3D(origin=Vec3f.forward(), direction=Vec3f.backwards()),
            Ray3D(origin=Vec3f.left(), direction=Vec3f.right()),
            Ray3D(origin=Vec3f.down(), direction=Vec3f.up()),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f.forward()),
            Ray3D(origin=Vec3f([-1, -1, -1]), direction=Vec3f([1, 1, 1])),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f([0, 1, 1]))
        ]

        box = Box3D()

        for ray in hit_rays:
            self.assertTrue(ray.intersects(box),
                            "The ray originating at %s and travelling in the "
                            "direction %s should intersect the cube centered "
                            "at %s with the dimensions %s." % (ray.origin,
                                                               ray.direction,
                                                               box.centroid,
                                                               box.dimensions))

        miss_rays = [
            Ray3D(origin=Vec3f.right(), direction=Vec3f.right()),
            Ray3D(origin=Vec3f.up(), direction=Vec3f.up()),
            Ray3D(origin=Vec3f.forward(), direction=Vec3f.forward()),
            Ray3D(origin=Vec3f.left(), direction=Vec3f.left()),
            Ray3D(origin=Vec3f.down(), direction=Vec3f.down()),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f.backwards()),
            Ray3D(origin=Vec3f([-1, -1, -1]), direction=Vec3f([-1, -1, -1])),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f([0, -1, -1])),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f([0.01, 0.01, -1])),
            Ray3D(origin=Vec3f.backwards(), direction=Vec3f.backwards()),
        ]

        for ray in miss_rays:
            self.assertFalse(ray.intersects(box),
                             "The ray originating at %s and travelling in the "
                             "direction %s should NOT intersect the cube "
                             "centered at %s with the dimensions %s."
                             % (ray.origin, ray.direction,
                                box.centroid, box.dimensions))

    def test_ray_intersection_thin_box(self):
        ray = Ray3D(origin=Vec3f.backwards(), direction=Vec3f.forward())
        box = Box3D(vmin=Vec3f([-0.5, -0.5, 0]), vmax=Vec3f([0.5, 0.5, 0]))

        self.assertTrue(ray.intersects(box))
        self.assertEqual(ray.find_intersection(box), (1, 1))


if __name__ == '__main__':
    unittest.main()
