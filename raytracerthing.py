import numpy as np


from raytracing import Box3D, Vec3f, Ray3D


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
            self.pixel_values = np.array(pixel_values)

            assert self.pixel_values.shape == (n_rows, n_cols)
        else:
            self.pixel_values = np.random.uniform(low=0.0, high=1.0,
                                                  size=(n_rows, n_cols))

        self.pixel_size = pixel_size

        self.pixel_centers = []
        self.pixel_extents = []

        self.calculate_pixel_centers_and_extents()

    def calculate_pixel_centers_and_extents(self):
        """For each pixel, calculate the center point and the extents (top left and bottom right corners)."""
        for row in range(self.n_rows):
            self.pixel_centers.append([])
            self.pixel_extents.append([])

            for col in range(self.n_cols):
                pixel_x_min_extent = self.top_left.x + col * self.pixel_size
                pixel_x_max_extent = pixel_x_min_extent + self.pixel_size
                pixel_center_x = 0.5 * (pixel_x_min_extent + pixel_x_max_extent)

                pixel_y_min_extent = self.top_left.y - row * self.pixel_size
                pixel_y_max_extent = pixel_y_min_extent - self.pixel_size
                pixel_center_y = 0.5 * (pixel_y_min_extent + pixel_y_max_extent)

                pixel_min_extent = Vec3f([pixel_x_min_extent, pixel_y_min_extent, self.z])
                pixel_max_extent = Vec3f([pixel_x_max_extent, pixel_y_max_extent, self.z])
                pixel_center = Vec3f([pixel_center_x, pixel_center_y, self.z])

                self.pixel_centers[row].append(pixel_center)
                self.pixel_extents[row].append((pixel_min_extent, pixel_max_extent))

    @property
    def shape(self):
        """Get the shape of the pixel grid.

        Returns:
            A 2-tuple consisting of the grid height and width, in that order.
        """
        return self.pixel_values.shape

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
        row, col = self.to_grid_coords(x, y)

        return self.pixel_values[row, col]

    def to_grid_coords(self, x, y):
        """Convert world coordinates (only x and y) to grid coordinates (i.e. pixel array indices.

        It assumed that the point is contained within the pixel grid.

        Any points that coincide with the border between two pixels will cause the returned column to be that of the
        pixel located to the right

        Arguments:
            x: The x coordinate to convert.
            y: The y coordinate to convert.

        Returns:
            A 2-tuple that contains the row and column position of the pixel in the pixel grid.
        """
        # align bottom edge of grid to y=0 and flip y so that it increases in the same direction as the grid rows
        # (i.e. so that y increases in the negative y direction, starting from the top of the grid
        # (y = 0.5 * self.height).
        row = -y + 0.5 * self.height
        row = row / self.pixel_size  # convert from units of distance to number of pixels from bottom of grid.
        row = min(row, self.n_rows - 1)
        row = int(row)

        col = x + 0.5 * self.width  # align left edge of grid to x=0
        col = col / self.pixel_size  # convert from units of distance to number of pixels from LHS of grid.
        col = min(col, self.n_cols - 1)
        col = int(col)

        return row, col

    def __getitem__(self, item):
        """
        Get the value of a pixel in the pixel grid.

        Returns:
            the item in the pixel array at the location [y, x].
        """
        return self.pixel_values.__getitem__(item)

    def __str__(self):
        return str(self.pixel_values)


class RayTracerThing:
    """This thing does some stuff."""

    def __init__(self, input_shape, output_shape, n_layers=0,
                 hidden_layer_shape=None):
        """Create a ray tracer thing (need to think of a better name).

        Arguments:
            input_shape: The shape of the input image.
            output_shape: The shape of the detector array.
            n_layers: The number of 'hidden' layers to add.
            hidden_layer_shape: The shape of the 'hidden' layers. If set to
                                None, `hidden_layer_shape` defaults to
                                `input_shape`.
        """
        if hidden_layer_shape is None:
            hidden_layer_shape = input_shape

        self.input_shape = input_shape
        self.layer_shape = hidden_layer_shape
        self.output_shape = output_shape
        self.n_layers = n_layers

        self.output_layer = PixelGrid(*output_shape, z=0)
        self.hidden_layers = [PixelGrid(*hidden_layer_shape, z=1 + n)
                              for n in range(n_layers)]
        self.input_layer = PixelGrid(*input_shape, z=n_layers + 1)

        self.layer_after_input = self.hidden_layers[0] if len(self.hidden_layers) > 0 else self.input_layer

    def enable_full_transparency(self):
        """Enable full transparency on each hidden layer.

        Set each pixel in every hidden layer to have a value of 1, meaning that
        any light that passes through the hidden layers is unhindered and does
        not lose any intensity.

        Mainly useful for testing.
        """
        ones = np.ones(self.layer_shape)

        for layer in self.hidden_layers:
            layer.pixel_values = ones

    def enable_full_opacity(self):
        """Enable full opacity on each hidden layer.

        Set each pixel in every hidden layer to have a value of 0, meaning that
        any light that tries to pass through the hidden layers is completely
        blocked.

        Mainly useful for testing.
        """
        zeros = np.zeros(self.layer_shape)

        for layer in self.hidden_layers:
            layer.pixel_values = zeros

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

        self.input_layer.pixel_values = X
        output = np.zeros(shape=self.output_shape)

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                origin = self.output_layer.pixel_centers[row][col]

                for rows in self.layer_after_input.pixel_centers:
                    for target in rows:
                        direction = target - origin
                        ray = Ray3D(origin, direction)
                        pixel_value = self.input_layer.hit_value(ray)

                        if pixel_value is not None:
                            for layer in reversed(self.hidden_layers):
                                pixel_value *= layer.hit_value(ray)

                            output[row][col] += pixel_value

        return output
