import numpy as np
import tensorflow as tf

from raytracing import Box3D, Vec3f, Ray3D

tf.enable_eager_execution()


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
        #
        # for row in range(self.pixel_values.shape[0]):
        #     for col in range(self.pixel_values.shape[1]):
        #         pixel_values[row][col] = tf.

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


        Returns: the value of the pixel the ray intersects if such a pixel
                 exists, otherwise returns None if there is no intersection.
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
        """Convert world coordinates (only x and y) to grid coordinates
        (i.e. pixel array indices.

        It assumed that the point is contained within the pixel grid.

        Any points that coincide with the border between two pixels will cause
        the returned column to be that of the pixel located to the right.

        Arguments:
            x: The x coordinate to convert.
            y: The y coordinate to convert.

        Returns: A 2-tuple that contains the row and column position of the
        pixel in the pixel grid.
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

        Returns: the item in the pixel array at the location [y, x].
        """
        return self.pixel_values.__getitem__(item)

    def __str__(self):
        return str(self.pixel_values)


class Activations:
    """Implements a couple of activation functions."""

    @staticmethod
    def identity(x):
        """Apply the identity activation function.

        Arguments:
            x: the input.

        Returns: the input without any modifications.
        """
        return x

    @staticmethod
    def softmax(x):
        """Apply the softmax activation function to the input.

        Arguments:
            z: the input.

        Returns: the input transformed with the softmax function.
        """
        z = tf.exp(x)

        return z / tf.reduce_sum(z, axis=0)

    @staticmethod
    def sigmoid(x):
        z = tf.exp(x)
        return z / (z + 1)


class RayTracerThing:
    """This thing does some stuff."""

    def __init__(self, input_shape, n_layers=0,
                 hidden_layer_shape=None, activation_func=Activations.identity):
        """Create a ray tracer thing (need to think of a better name).

        Arguments:
            input_shape: The shape of the input image.
            n_layers: The number of 'hidden' layers to add.
            hidden_layer_shape: The shape of the 'hidden' layers. If set to
                                None, `hidden_layer_shape` defaults to
                                `input_shape`.
        """
        if hidden_layer_shape is None:
            hidden_layer_shape = input_shape

        self.input_shape = input_shape
        self.layer_shape = hidden_layer_shape
        self.output_shape = (1, 1)
        self.n_layers = n_layers

        self.output_layer = PixelGrid(*self.output_shape, z=0)
        self.hidden_layers = [PixelGrid(*hidden_layer_shape, z=1 + n)
                              for n in range(n_layers)]
        self.input_layer = PixelGrid(*input_shape, z=n_layers + 1)

        self.ray_grid_intersections = self._find_ray_grid_intersections()
        self.activation = activation_func

    def _find_ray_grid_intersections(self):
        """Find the grid coordinates for where each ray cast during a forward
        pass would intersect each of the hidden layers.

        For each pixel in the output layer (detector array), a ray is cast to
        each pixel in the input layer. For each hidden layer (a pixel grid in
        between the input and output layers), the grid coordinates (row, col)
        are recorded for where the ray intersects that layer. This is
        essentially pre-calculating the ray paths.

        Returns: A 5-D array where each element is a grid coordinate (row, col)
                 for where a ray originating at the pixel at (*m, n*) cast to
                 the pixel at (*i, j*) intersects the hidden layer *l*. The
                 array is of the dimensions (M, N, I, J, L), where M and N are
                 the row and column components of the output shape, I and J the
                 row and column components of the input shape, and L is the
                 number of hidden layers.
        """
        ray_grid_intersections = []

        for row in range(self.output_layer.n_rows):
            ray_grid_intersections.append([])

            for col in range(self.output_layer.n_cols):
                origin = self.output_layer.pixel_centers[row][col]

                ray_grid_intersections[row].append([])

                for input_row in self.input_layer.pixel_centers:
                    ray_grid_intersections[row][col].append([])

                    for target in input_row:
                        ray_grid_intersections[row][col][-1].append([])

                        direction = target - origin
                        ray = Ray3D(origin, direction)

                        intersections = []

                        for layer in self.hidden_layers:
                            intersection_t = layer.bounding_box.find_intersection(ray)

                            if intersection_t is None:
                                break

                            intersection_point = ray.get_point(intersection_t[0])
                            grid_coords = layer.to_grid_coords(intersection_point.x, intersection_point.y)

                            intersections += [grid_coords]
                        else:  # Ray intersects all layers between input and output layers.
                            ray_grid_intersections[row][col][-1][-1] = intersections

        return ray_grid_intersections

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

    @tf.contrib.eager.defun
    def forward(self, x):
        """Perform a 'forward pass' of the ray tracer thing.

        Arguments:
            x: The input image, must be the same shape as `input_shape`.

        Returns:
            2-D array of values that is the same shape as `output_shape`.

        Raises:
            AssertionError: if the shape of `X` does not match `input_shape`.
        """
        assert x.shape == self.input_shape, "Expected input to be of the " \
                                            "shape %s, instead got %s." \
                                            % (self.input_shape, x.shape)

        # x = tf.cast(x, tf.float32)

        ray_values = []

        for input_row in range(self.input_layer.n_rows):
            for input_col in range(self.input_layer.n_cols):
                transparency_values = []
                intersection_grid_coords = self.ray_grid_intersections[0][0][input_row][input_col]

                for layer, (grid_row, grid_col) in zip(self.hidden_layers, intersection_grid_coords):
                    transparency = layer.pixel_values[grid_row][grid_col]
                    transparency_values.append(transparency)

                transparency = tf.reduce_prod(transparency_values)
                ray_value = tf.multiply(transparency, x[input_row][input_col])

                ray_values.append(ray_value)

        output = tf.reduce_sum(ray_values)

        return self.activation(output)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.datasets import load_digits

    np.random.seed(42)

    digits = load_digits()
    digits.keys()

    y = digits['target']

    X = digits['images']
    X = X[y < 2]
    X = X / X.max()
    N = X.shape[0]
    image_shape = X.shape[1:]

    print(N, image_shape)

    y = y[y < 2]

    print(y[:5])

    layer_shape = image_shape

    clf = RayTracerThing(input_shape=image_shape,
                         hidden_layer_shape=layer_shape, n_layers=3,
                         activation_func=Activations.sigmoid)

    outputs = []

    for i, image in enumerate(X):
        print('\rImage %d of %d' % (i + 1, N), end='')
        outputs.append(clf.forward(image))

    print()

    fig, axes = plt.subplots(5, 2, figsize=(9, 15))
    axes = axes.ravel()

    for ax, image, expected, actual in zip(axes, X[:10], y[:10], outputs[:10]):
        sns.heatmap(image, vmin=0.0, vmax=1.0, cmap='gray', ax=ax)
        ax.set_axis_off()

        actual = 0 if actual < 0.5 else 1

        color = 'green' if expected == actual else 'red'
        ax.set_title('Predicted %d' % actual, color=color)

    plt.show()
