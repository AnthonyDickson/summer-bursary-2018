import numpy as np


class Vec3D(np.ndarray):
    @staticmethod
    def zero():
        return Vec3D([0, 0, 0])

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        assert obj.shape == (3,), "Vec3D expects an 1-D input of shape (1, ), instead got an input of shape %s" % obj.shape

        return obj

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    def normalise(self):
        if np.array_equal(self, Vec3D.zero()):
            return self

        return self / np.sqrt(np.square(self).sum())


class Ray:
    def __init__(self, origin=None, direction=None):
        if origin is None:
            origin = Vec3D.zero()

        if direction is None:
            direction = Vec3D.zero()

        assert origin.shape == (3, )
        assert direction.shape == (3, )

        direction = direction.normalise()

        self.origin = origin
        self.direction = direction

    def get_point(self, t=0):
        return self.origin + t * self.direction


class RayTracerThing:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, X):
        assert X.shape == self.input_shape
