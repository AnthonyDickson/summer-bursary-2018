import numpy as np


from raytracing import Box3D, Vec3f


class PixelGrid:
    """A 2D grid of pixels in 3D space."""

    def __init__(self, n_rows=1, n_cols=1, z=0, pixel_size=1, pixel_values=None):
        """Create a pixel grid.

        Arguments:
            n_rows: How many rows the grid should have.
            n_cols: How many columns the grid should have.
            z: How far forward the grid is positioned.
            pixel_size: The size (both width and height) of the pixels in the grid.
            pixel_values: An array with values to fill the grid with. This must
                          be of the shape (`n_rows`, `n_cols`). If set to None,
                          the grid is filled with zeroes.
        """
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.z = z

        self.height = n_rows * pixel_size
        self.width = n_cols * pixel_size

        self.top_left = Vec3f([-0.5 * self.width, 0.5 * self.height, z])
        self.top_right = Vec3f([0.5 * self.width, 0.5 * self.height, z])
        self.bottom_left = Vec3f([-0.5 * self.width, -0.5 * self.height, z])
        self.bottom_right = Vec3f([0.5 * self.width, -0.5 * self.height, z])

        self.origin = 0.5 * (self.top_left + self.bottom_right)

        self.bounding_box = Box3D(vmin=self.bottom_left, vmax=self.top_right)

        if pixel_values is not None:
            self.pixels = np.array(pixel_values)

            assert self.pixels.shape == (n_rows, n_cols)
        else:
            self.pixels = np.zeros((n_rows, n_cols))

        self.pixel_size = pixel_size

    @property
    def shape(self):
        """Get the shape of the pixel grid.

        Returns:
            A 2-tuple consisting of the grid height and width, in that order.
        """
        return self.pixels.shape

    def hit_value(self, ray):
        """Find the value of the pixel that a given ray intersects with.

        Arguments:
            ray: The ray to intersect the grid with.


        Returns:
            the value of the pixel the ray intersects if such a pixel exists,
            otherwise returns None if there is no intersection.
        """
        intersection_t = self.bounding_box.find_intersection(ray)

        if intersection_t is None:
            return None

        # Since the pixel grid has a depth of zero, both t values will be the
        # same, so we can discard the second value.
        intersection_t = intersection_t[0]
        intersection_point = ray.get_point(intersection_t)
        x, y, _ = intersection_point
        row, col = self.xy_to_grid_coords(x, y)

        return self.pixels[row, col]

    def xy_to_grid_coords(self, x, y):
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                x0 = self.top_left.x + col * self.pixel_size
                x1 = x0 + self.pixel_size

                x0, x1 = min(x0, x1), max(x0, x1)

                y0 = self.top_left.y - row * self.pixel_size
                y1 = y0 - self.pixel_size
                y0, y1 = min(y0, y1), max(y0, y1)

                if x0 <= x <= x1 and y0 <= y <= y1:
                    return row, col

        return None

        # col = int(x - self.top_left.x)
        # row = int(y + self.top_left.y)
        #
        # col = col // self.n_cols
        # row = row % self.n_rows
        #
        # return row, col

    def __getitem__(self, item):
        """
        Get the value of a pixel in the pixel grid.

        Returns:
            the item in the pixel array at the location [y, x].
        """
        return self.pixels.__getitem__(item)

    def __str__(self):
        return str(self.pixels)


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
