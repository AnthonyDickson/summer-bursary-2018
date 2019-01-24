import numpy as np


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
