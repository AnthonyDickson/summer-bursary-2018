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


class Ray:
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

        """
        if origin is None:
            origin = Vec3f.zero()

        if direction is None:
            direction = Vec3f.zero()

        assert origin.shape == (3, )
        assert direction.shape == (3, )

        direction = direction.normalise()

        self.origin = origin
        self.direction = direction
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

    def intersects_plane(self, plane):
        """Check if the ray intersects a plane.

        Based on code from: https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection

        Arguments:
            plane: The plane to check intersection with.

        Returns:
            True if the plane and the object intersect, otherwise False.
        """
        denom = self.direction.dot(plane.norm)
        parallel = denom < self.epsilon

        if parallel:
            return (plane.origin - self.origin).dot(plane.norm) < self.epsilon
        else:
            return True


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
            assert vector.shape == (3, ), "Plane3D expects a vector with " \
                                          "the shape (3, ), instead got %s." \
                                          % vector.shape

        assert np.count_nonzero(norm) > 0, "At least one element of `norm` " \
                                           "must be non-zero."

        self.origin = origin
        self.norm = norm.normalise()
        self.d = origin.dot(norm)

    def point_intersects(self, point):
        """Check if a point intersects (i.e. lies on) a plane.

        Arguments:
            point: The point in 3-D space to check intersection for.

        Returns:
            True if the point lies on the plane, otherwise False.
        """

        return self.origin.dot(point) == self.d


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
