"""This module provides access to two simple 3D objects (planes and
axis-aligned bounding boxes) and parametric rays. The objects also provide the
functionality needed for testing intersection between these objects and rays.
"""

import numpy as np


class Vec3f(np.ndarray):
    """Represents a row vector with three elements: x, y, and z."""

    @staticmethod
    def zero():
        """Create a zero vector.

        Returns:
            a zero vector with three elements.
        """
        return Vec3f([0, 0, 0])

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        assert obj.shape == (3,), "Vec3D expects an 1-D input of " \
                                  "shape (3, ), instead got an input of " \
                                  "shape %s" % obj.shape

        return obj.astype(np.float64)

    @property
    def x(self):
        """The first element of the vector, x.

        Returns:
            the first element of the vector.
        """
        return self[0]

    @property
    def y(self):
        """The second element of the vector, y.

        Returns:
            the second element of the vector.
        """
        return self[1]

    @property
    def z(self):
        """The third element of the vector, z.

        Returns: the third element of the vector.
        """
        return self[2]

    def normalise(self):
        """Normalise the vector.

        Returns:
            the corresponding unit vector, or a zero vector if the
            initial vector was also a zero vector.
        """
        if np.array_equal(self, Vec3f.zero()):
            return self

        return self / np.sqrt(np.square(self).sum())


class Object3D:
    """Interface that defines a few common methods for 3D objects that need to
    be support ray tracing, in particular ray intersection testing."""

    def contains(self, point):
        """Check if the object contains a point.

        Arguments:
            point: The point to check.

        Returns:
            True if the point is contained within the object, otherwise False.
        """
        raise NotImplementedError

    def find_intersection(self, ray):
        """Find the value of t for which a parametrically defined ray
        intersects the object.

        Arguments:
            ray: The ray to find the intersection with.

        Returns:
            The value of t for which the ray intersects the object if such a
            value exists, otherwise None if there is no intersection or t is
            negative.
        """
        raise NotImplementedError

    def intersects(self, ray):
        """Check if a ray intersects the object.

        Arguments:
            ray: The object to test for intersection with.

        Returns:
            True if the ray and the object intersect, otherwise False.
        """
        return self.find_intersection(ray) is not None


class Plane3D(Object3D):
    """A plane in 3-D space defined by two points."""

    def __init__(self, origin=None, norm=None):
        """Create a plane in 3-D space.

        Arguments:
            origin: The point in space from where the plane originates. If set
                    to None then origin is set to the point (0, 0, 0).
            norm: The vector normal to the plane, (i.e. the vector pointing
                  'upwards' perpendicularly from the plane). If set to None
                  then the norm is set to point in the positive z direction,
                  (0, 0, 1).

        Raises:
            AssertionError: if none of the elements in `norm` are non-zero.
        """
        if origin is None:
            origin = Vec3f.zero()

        if norm is None:
            norm = Vec3f([0, 0, 1])

        for vector in [origin, norm]:
            assert vector.shape == (3,), "Plane3D expects a vector with " \
                                         "the shape (3, ), instead got %s." \
                                         % vector.shape

        assert np.count_nonzero(norm) > 0, "At least one element of `norm` " \
                                           "must be non-zero."

        self.origin = origin
        self.norm = norm.normalise()
        self.d = origin.dot(norm)

    def contains(self, point):
        """Check if a plane contains a point.

        Arguments:
            point: The point in 3-D space to check intersection for.

        Returns:
            True if the point lies on the plane, otherwise False.
        """
        return self.origin.dot(point) == self.d

    def find_intersection(self, ray):
        """Find the intersection between the plane and a ray.

        Arguments:
            ray: The find to find the intersection with.

        Returns:
            the value of t for which the ray intersects the plane (the single
            value t=0 is returned if the ray contained in the plane) if there
            is an intersection, None if the intersection occurs 'behind' the
            ray (i.e. t is negative) or the ray is parallel to the plane.
        """
        num = (self.origin - ray.origin).dot(self.norm)
        denom = ray.direction.dot(self.norm)
        parallel = abs(denom) < ray.epsilon

        if parallel:
            return 0 if num < ray.epsilon else None
        else:
            t = num / denom

            return t if t >= 0 else None


class Box3D(Object3D):
    """Represents a axis-aligned bounding box (AABB) in 3-D space defined by
    two points, the minimum extent and maximum extent.

    Based on code from: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    """

    def __init__(self, vmin=None, vmax=None):
        """Create a bounding box defined by two points in space (the minimum
        and maximum extents, i.e. a set of opposite corners in a cube).

        If both parameters are left at their default values then a unit cube
        centered around the origin is created.

        Arguments:
            vmin: The vector describing the minimum extent of the cube. If set
                  to None then `vmin` is set to the vector (-0.5, -0.5, -0.5).
            vmin: The vector describing the maximum extent of the cube. If set
                  to None then `vmax` is set to the vector (0.5, 0.5, 0.5).

        Raises:
            AssertionError: if either of the input vectors are not the correct
                            shape, (3, ).
        """
        if vmin is None:
            vmin = Vec3f([-0.5, -0.5, -0.5])

        if vmax is None:
            vmax = Vec3f([0.5, 0.5, 0.5])

        for vector in [vmin, vmax]:
            assert vector.shape == (3,), "Box3D expects a vector with " \
                                         "the shape (3, ), instead got %s." \
                                         % vector.shape

        self.vmin = vmin
        self.vmax = vmax
        self.dimensions = vmax - vmin
        self.centroid = (1 / len(vmin)) * (vmax + vmin)

    def contains(self, point):
        """Check if the bounding box contains a point.

        Arguments:
            point: The point to check.

        Returns:
            True if the bounding box contains the point, False otherwise.
        """
        return (self.vmin.x <= point.x <= self.vmax.x) and \
               (self.vmin.y <= point.y <= self.vmax.y) and \
               (self.vmin.z <= point.z <= self.vmax.z)

    def find_intersection(self, ray):
        """Find the intersection(s) between the bounding box and a ray.

        Arguments:
            ray: The ray to find the intersection(s) with.

        Returns:
            If there is an intersection a 2-tuple of values representing the
            values of t for which the ray intersects the box, otherwise None.
        """
        t_near = float('-inf')
        t_far = float('inf')

        for i in range(3):
            if np.abs(ray.inverse_direction[i]) < ray.epsilon:
                if ray.origin[i] < self.vmin[i] or ray.origin[i] > self.vmax[i]:
                    return None
            else:
                t0 = (self.vmin[i] - ray.origin[i]) * ray.inverse_direction[i]
                t1 = (self.vmax[i] - ray.origin[i]) * ray.inverse_direction[i]

                if t0 > t1:
                    t0, t1 = t1, t0

                if t0 > t_near:
                    t_near = t0

                if t1 < t_far:
                    t_far = t1

                if t_near > t_far or t_far < 0:
                    return None

        return t_near, t_far

    def __mul__(self, other):
        vmin = self.vmin * other
        vmax = self.vmax * other

        return Box3D(vmin, vmax)

    def __rmul__(self, other):
        return self.__mul__(other)


class Ray3D:
    """Represents a ray object used in ray tracing."""

    def __init__(self, origin=None, direction=None):
        """Create a ray.

        Arguments:
            origin: The point in space from where the ray originates. If set to
                    None then origin is set to the point (0, 0, 0).
            direction: The direction that the ray is travelling in. If set to
                       None then the direction is set to point in the positive
                       z direction, (0, 0, 1). The direction will be
                       normalised.

        Raises:
            AssertionError: if either of the input vectors are not the correct
                            shape, (3, ).
        """
        if origin is None:
            origin = Vec3f.zero()

        if direction is None:
            direction = Vec3f.zero()

        assert origin.shape == (3,)
        assert direction.shape == (3,)

        direction = direction.normalise()

        self.origin = origin
        self.direction = direction
        # use np.divide to avoid division by zero warnings.
        self.inverse_direction = np.divide(1, direction,
                                           out=np.zeros_like(direction),
                                           where=direction != 0)
        self.epsilon = np.finfo(type(self.origin.x)).eps

    def get_point(self, t=0):
        """Get the point along the ray for a given value of t.

        Returns:
            the point along the ray for the given value of t.
        """
        return self.origin + t * self.direction

    def intersects(self, other):
        """Check if the ray intersects an object.

        Arguments:
            other: The object to check intersection with.

        Returns:
            True if the ray and the object intersect, otherwise False.

        Raises:
            AttributeError: if the object `other` does not support the method
                            `intersects`.
        """
        return other.intersects(self)

    def find_intersection(self, other):
        """Find the value(s) of t for which the ray intersects the object.

        Arguments:
            other: The object to find the intersection with.

        Returns:
            The value(s) of t for which the ray intersects the object, None if
            there is no intersection or the value of t is negative.

        Raises:
            AttributeError: if the object `other` does not support the method
                            `find_intersection`.
        """
        return other.find_intersection(self)
