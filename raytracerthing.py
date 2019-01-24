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


class Plane3D:
    """A plane in 3-D space."""

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


class Box3D:
    """Represents a axis-aligned bounding box (AABB) in 3-D space.

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
        """Check if a bounding box contains a point.

        Arguments:
            point: The point to check.

        Returns:
            True if the bounding box contains the point, False otherwise.
        """
        return (self.vmin.x <= point.x <= self.vmax.x) and \
               (self.vmin.y <= point.y <= self.vmax.y) and \
               (self.vmin.z <= point.z <= self.vmax.z)

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
        """Check if the ray intersects another object.

        Arguments:
            other: The object to check intersection with.

        Returns:
            True if the ray and the object intersect, otherwise False.
        """
        if isinstance(other, Plane3D):
            return self.intersects_plane(other)
        elif isinstance(other, Box3D):
            return self.intersects_box(other)
        else:
            return NotImplemented

    def intersects_plane(self, plane):
        """Check if the ray intersects a plane.

        Arguments:
            plane: The plane to check intersection with.

        Returns:
            True if the ray and the plane intersect, otherwise False.
        """
        denom = self.direction.dot(plane.norm)
        parallel = denom < self.epsilon

        if parallel:
            return (plane.origin - self.origin).dot(plane.norm) < self.epsilon
        else:
            return True

    def intersects_box(self, box):
        """Check if the ray intersects a box.

        Arguments:
            box: The box to check intersection with.

        Returns:
            True if the ray and the box intersect, otherwise False.
        """
        return self.box_intersection(box) is not None

    def box_intersection(self, box):
        """Find the intersection(s) between the ray and a box (AABB).

        Arguments:
            box: The box to find the intersection(s) with.

        Returns:
            If there is an intersection a 2-tuple of values representing the values of t for which the ray intersects
            the box, otherwise None.
        """
        t_near = float('-inf')
        t_far = float('inf')

        for i in range(3):
            if np.abs(self.inverse_direction[i]) < self.epsilon:
                if self.origin[i] < box.vmin[i] or self.origin[i] > box.vmax[i]:
                    return None
            else:
                t0 = (box.vmin[i] - self.origin[i]) * self.inverse_direction[i]
                t1 = (box.vmax[i] - self.origin[i]) * self.inverse_direction[i]

                if t0 > t1:
                    t0, t1 = t1, t0

                if t0 > t_near:
                    t_near = t0

                if t1 < t_far:
                    t_far = t1

                if t_near > t_far or t_far < 0:
                    return None

        return t_near, t_far


class RayTracerThing:
    """This thing does some stuff."""

    def __init__(self, input_shape, output_shape):
        """Create a ray tracer thing (need to think of a better name).

        Arguments:
            input_shape: The shape of the input image.
            output_shape: The shape of the detector array.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        """Perform a 'forward pass' of the ray tracer thing.

        Arguments:
            X: The input image, must be the same shape as `input_shape`.

        Returns:
            2-D array of values that is the same shape as `output_shape`.

        Raises:
            AssertionError: if the shape of `X` does not match `input_shape`.
        """
        assert X.shape == self.input_shape, "Expected input to be of the " \
                                            "shape %s, instead got %s." \
                                            % (self.input_shape, X.shape)

        output = np.zeros(shape=self.output_shape)
