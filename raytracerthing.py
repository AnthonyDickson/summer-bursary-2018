import numpy as np
import torch

from pixelgrid import PixelGrid
from raytracing import Ray3D

torch.set_default_dtype(torch.float64)


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
        z = torch.exp(x)

        return z / torch.sum(z, axis=0)

    @staticmethod
    def sigmoid(x):
        z = torch.exp(x)
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
        self.activation = activation_func

        self.input_layer = PixelGrid(*input_shape, z=0)
        self.hidden_layers = [PixelGrid(*hidden_layer_shape, z=1 + n)
                              for n in range(n_layers)]
        self.output_layer = PixelGrid(*self.output_shape, z=n_layers + 1)

        self._setup()

    def _setup(self):
        """Prepare the model for training.

        This should be called before training and if the pixel values are
        modified manually to ensure that the weights and pixel values are in
        sync.
        """
        self.ray_grid_intersections = self._find_ray_grid_intersections()
        self.W = self._get_W()
        self.grid_W_map = self._get_grid_W_map()

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

                for input_row in range(self.input_layer.n_rows):
                    ray_grid_intersections[row][col].append([])

                    for input_col in range(self.input_layer.n_cols):
                        ray_grid_intersections[row][col][input_row].append([])
                        target = self.input_layer.pixel_centers[input_row][input_col]

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

    def _get_W(self):
        """Generate the weight matrices based on the pixel values of the hidden layers and their ray intersections.

        Returns: A list of weight matrices that is the same shape as the input.
        """
        W = [torch.zeros(self.input_shape) for _ in range(self.n_layers)]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                for input_row in range(self.input_layer.n_rows):
                    for input_col in range(self.input_layer.n_cols):
                        intersections = self.ray_grid_intersections[row][col][input_row][input_col]

                        for layer in range(self.n_layers):
                            W[layer][input_row, input_col] = self.hidden_layers[layer].pixel_values[
                                intersections[layer]]

        for w in W:
            w.requires_grad_(True)
            w.retain_grad()

        return W

    def _get_grid_W_map(self):
        """Generate a mapping between the pixels in the hidden layers and the elements in the weight matrices.

        Some rays intersect the same pixel as other rays. This means that they
        share the same 'weights' for that layer. To make computations easier
        these values are extracted and arranged in a matrix where each element
        represents the pixel for where the given ray intersects the given
        layer. This means there are possibly multiple elements that share the
        same pixel, and this must be taken into account when doing
        back-propagation. Thus, this mapping enables easy selection of weights
        that share the same pixel.

        Returns: a mapping between the pixels in the hidden layers and the elements in the weight matrices.
        """
        grid_W_map = [{} for _ in range(self.n_layers)]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                for input_row in range(self.input_layer.n_rows):
                    for input_col in range(self.input_layer.n_cols):
                        intersections = self.ray_grid_intersections[row][col][input_row][input_col]

                        for layer in range(self.n_layers):
                            try:
                                if input_row not in grid_W_map[layer][intersections[layer]]['row_slice']:
                                    grid_W_map[layer][intersections[layer]]['row_slice'].append(input_row)
                                    grid_W_map[layer][intersections[layer]]['col_slice'].append(input_col)
                            except KeyError:
                                grid_W_map[layer][intersections[layer]] = {'row_slice': [input_row],
                                                                           'col_slice': [input_col]}

        return grid_W_map

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

        self._setup()

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

        self._setup()

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

        output = torch.tensor(x)

        for w in self.W:
            output = output * w

        output = torch.sum(output)  # (1, 2))

        return self.activation(output)

    def zero_grad(self):
        """Prepare the graph for the next epoch.

        PyTorch doesn't allow you to call `backward()` on the same graph twice,
        so it is necessary to manually detach tensors from the graph and zero
        gradients.
        """
        for w in self.W:
            w.detach_()
            w.requires_grad_(True)
            w.retain_grad()

            if w.grad is not None:
                w.grad.detach_()
                w.grad.zero_()

    def broadcast_pixel_values(self):
        """Update the weights that share the same pixel so that they have the same value.

        Some elements in the weight matrices refer to the same pixel in the
        pixel grid, however the pixel value is represented as multiple entries
        in the weights matrices and are trained with back-propagation
        separately. In reality, a single pixel can only take on a single value,
        so this must be resolved by deciding on a single value for the pixel
        and assigning this value to each of the matrix elements that refer to
        the pixel.
        """

        for layer in range(self.n_layers):
            # The weights represent transparency values so they should be
            # clamped to the interval [0.0, 1.0].
            self.W[layer] = torch.clamp(self.W[layer], 0.0, 1.0)

            # Adjust for pixels that are represented as separate values in the
            # weight matrices but are actually a single entity.
            for grid_coord in self.grid_W_map[layer]:
                row_slice = self.grid_W_map[layer][grid_coord]['row_slice']
                col_slice = self.grid_W_map[layer][grid_coord]['col_slice']

                mean_pixel_value = self.W[layer][row_slice, col_slice].mean()
                self.W[layer][row_slice, col_slice] = mean_pixel_value

                # Keep the original pixel array up to date.
                self.hidden_layers[layer].pixel_values[row_slice, col_slice] = mean_pixel_value
