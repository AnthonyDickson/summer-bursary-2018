"""This module defines a classifier that has an interface similar to
scikit-learn models.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

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
            x: the input.

        Returns: the input transformed with the softmax function.
        """
        return torch.softmax(x, dim=1)

    @staticmethod
    def sigmoid(x, alpha=0):
        z = torch.exp(x - alpha)
        return z / (z + 1)


class Losses:
    """Implements a couple of loss functions."""

    @staticmethod
    def log_loss(predicted_prob, true_label):
        if not isinstance(true_label, torch.Tensor):
            true_label = torch.tensor(true_label)

        return torch.where(true_label == 1, -torch.log(predicted_prob), -torch.log(1 - predicted_prob))


def generate_minibatches(X, y, batch_size):
    """Split `X` and `y` into minibatches with `batch_size` elements each.

    Arguments:
        X: The feature data to split.
        y: The label data to split.
        batch_size: The size of the batches to generate.

    Yields: A 3-tuple containing the batch number, the X batch, and the y batch.
    """
    i_start = 0
    batch_number = 0

    while i_start < len(X):
        i_end = i_start + batch_size

        if i_end > len(X):
            i_end = len(X)

        yield batch_number, X[i_start:i_end], y[i_start:i_end]

        i_start = i_end
        batch_number = batch_number + 1


class RayTracerThing:
    """This thing does some stuff."""

    def __init__(self, input_shape, n_hidden_layers=0, hidden_layer_shape=None,
                 n_classes=2, activation_func=Activations.softmax,
                 loss_func=torch.nn.functional.cross_entropy, learning_rate=1,
                 pixel_density=8):
        """Create a ray tracer thing (need to think of a better name).

        Arguments:
            input_shape: The shape of the input image.
            n_hidden_layers: The number of 'hidden' layers to add.
            hidden_layer_shape: The shape of the 'hidden' layers. If set to
                                None, `hidden_layer_shape` defaults to
                                `input_shape`.
            n_classes: The number of classes the classifier should expect.
            activation_func: The activation to apply at the output layer.
            loss_func: The loss function to use.
            learning_rate: The learning rate.
            pixel_density: The pixel density of the hidden layers relative to
                           the input and output layers.
        """
        if hidden_layer_shape is None:
            hidden_layer_shape = input_shape

        assert hidden_layer_shape[0] >= input_shape[0] and \
               hidden_layer_shape[1] >= input_shape[1], \
            "The hidden layer dimensions should be at least as large as the " \
            "dimensions of the input layer. An input shape of %s was given " \
            "with a hidden layer shape of %s." % (input_shape, hidden_layer_shape)

        assert min(hidden_layer_shape) >= n_classes, \
            "The minimum dimension of the hidden layer shape should be no " \
            "less than the number of classes. There are %d classes specified " \
            "but hidden layer dimensions of %s were given." % (n_classes, hidden_layer_shape)

        hidden_layer_shape = (pixel_density * hidden_layer_shape[0],
                              pixel_density * hidden_layer_shape[1])

        self.input_shape = input_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.output_shape = (1, n_classes)
        self.n_classes = n_classes
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation_func
        self.loss = loss_func
        self.learning_rate = learning_rate

        self.input_layer = PixelGrid(*input_shape,
                                     pixel_size=pixel_density,
                                     z=0)
        self.hidden_layers = [PixelGrid(*hidden_layer_shape, z=1 + n)
                              for n in range(n_hidden_layers)]
        self.output_layer = PixelGrid(*self.output_shape,
                                      pixel_size=pixel_density,
                                      z=n_hidden_layers + 1)

        self.ray_grid_intersections = self._find_ray_grid_intersections()
        self.W = self._get_W()
        self.grid_W_map = self._get_grid_W_map()
        self.inverse_pixel_frequencies = self._get_inverse_pixel_frequencies()

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
        weights = [[None for _ in range(self.output_shape[1])] for _ in range(self.output_shape[0])]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                W = [torch.zeros(self.input_shape) for _ in range(self.n_hidden_layers)]

                for input_row in range(self.input_layer.n_rows):
                    for input_col in range(self.input_layer.n_cols):
                        intersections = self.ray_grid_intersections[row][col][input_row][input_col]

                        for layer in range(self.n_hidden_layers):
                            W[layer][input_row, input_col] = self.hidden_layers[layer].pixel_values[
                                intersections[layer]]

                for w in W:
                    w.requires_grad_(True)
                    w.retain_grad()

                weights[row][col] = W

        return weights

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
        grid_W_maps = [[None for _ in range(self.output_shape[1])] for _ in range(self.output_shape[0])]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                grid_W_map = [{} for _ in range(self.n_hidden_layers)]

                for input_row in range(self.input_layer.n_rows):
                    for input_col in range(self.input_layer.n_cols):
                        intersections = self.ray_grid_intersections[row][col][input_row][input_col]

                        for layer in range(self.n_hidden_layers):
                            try:
                                grid_W_map[layer][intersections[layer]]['row_slice'].append(input_row)
                                grid_W_map[layer][intersections[layer]]['col_slice'].append(input_col)
                            except KeyError:
                                grid_W_map[layer][intersections[layer]] = {'row_slice': [input_row],
                                                                           'col_slice': [input_col]}

                grid_W_maps[row][col] = grid_W_map

        return grid_W_maps

    def _get_inverse_pixel_frequencies(self):
        """Calculate the pixel frequencies (number of rays that intersect each
        pixel in the hidden layers) for each hidden layer.

        The pixel frequencies denote how many rays intersect a given pixel in a
        given hidden layer. These frequencies are used to average the weight
        matrices during training (see broadcast_pixel_values()).

        The inverse pixel frequencies are used as an optimisation since
        multiplication is typically faster than division, and averaging the
        weight matrices is something that happens frequently during training.

        Returns: a list of inverse pixel frequencies for each layer.
        """
        pixel_frequencies = [np.zeros(self.hidden_layer_shape) for _ in range(self.n_hidden_layers)]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                for layer in range(self.n_hidden_layers):
                    for grid_row, grid_col in self.grid_W_map[row][col][layer]:
                        row_slice = self.grid_W_map[row][col][layer][grid_row, grid_col]['row_slice']

                        pixel_frequencies[layer][grid_row][grid_col] += len(row_slice)

        inverse_pixel_frequencies = []

        for layer in range(self.n_hidden_layers):
            layer_pixel_frequencies = np.divide(pixel_frequencies[layer], 1,
                                                out=np.zeros_like(pixel_frequencies[layer]),
                                                where=pixel_frequencies[layer] != 0)
            inverse_pixel_frequencies.append(layer_pixel_frequencies)

        return inverse_pixel_frequencies

    def fit(self, X, y, val_data=0.2, n_epochs=100, batch_size=32,
            early_stopping=10, early_stopping_epsilon=1e-9,
            min_learning_rate=1e-4):
        """Fit the classifier.

        Arguments:
            X: The feature data.
            y: The label data.
            val_data: The data that should be used for validation. This can
                      be a tuple containing the validation X and y, or a ratio
                      or an integer indicating how many instances from `X` and
                      `y` should be used for validation.
            n_epochs: How many epochs to fit the model for.
            batch_size: The size of the minibatches to use. If set to -1, then
                        the batch size is set to the length of X.
            early_stopping: How many epochs to stop after if the loss has not
                            improved (i.e. decreased).
            early_stopping_epsilon: The minimum amount of improvement in loss
                                    before there is considered to be no
                                    improvement.
            min_learning_rate: The lowest value the learning rate should be
                               lowered to before stopping early.
        """
        assert len(X) == len(y), 'Inputs X and y should be the same length.'

        try:
            X_val, y_val = val_data
        except TypeError:  # val_data was not a tuple.
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_data)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
            y_val = torch.tensor(y_val)

        if batch_size == -1:
            batch_size = len(X)

        # early stopping stuff
        best_loss = float('inf')
        n_epochs_no_improvement = 0
        patience = early_stopping

        for epoch in range(n_epochs):
            epoch_loss = 0
            epoch_acc = 0

            for batch_number, X_batch, y_batch in generate_minibatches(X, y, batch_size):
                y_pred = self.predict_proba(X_batch)
                train_loss = self.loss(y_pred, y_batch).mean()
                train_loss.backward()

                train_accuracy = self.score(X_batch, y_batch)

                with torch.no_grad():
                    for row in range(self.output_layer.n_rows):
                        for col in range(self.output_layer.n_cols):
                            for layer in range(self.n_hidden_layers):
                                self.W[row][col][layer] -= self.learning_rate * self.W[row][col][layer].grad

                    self._broadcast_pixel_values()

                epoch_loss += train_loss
                epoch_acc += train_accuracy

                print('Epoch %d of %d - %d/%d - train_loss: %.4f - train_acc: %.4f'
                      % (epoch + 1, n_epochs,
                         batch_number * batch_size, len(X),
                         epoch_loss / (batch_number + 1), epoch_acc / (batch_number + 1)),
                      end='\r')

            with torch.no_grad():
                y_pred = self.predict_proba(X_val)
                val_loss = self.loss(y_pred, y_val).mean()
                val_accuracy = self.score(X_val, y_val)

            print('Epoch %d of %d - train_loss: %.4f - train_acc: %.4f - '
                  'val_loss: %.4f - val_acc: %.4f'
                  % (epoch + 1, n_epochs,
                     epoch_loss / (batch_number + 1), epoch_acc / (batch_number + 1),
                     val_loss, val_accuracy))

            if best_loss - val_loss > early_stopping_epsilon:
                best_loss = val_loss
                n_epochs_no_improvement = 0
            else:
                n_epochs_no_improvement += 1

            if n_epochs_no_improvement > patience:
                self.learning_rate *= 0.1
                n_epochs_no_improvement = 0

                if self.learning_rate >= min_learning_rate:
                    print('Decreasing learning rate to %.4f' % self.learning_rate)
                else:
                    print('\nStopping early.')
                    break

        print()

    def predict_proba(self, X):
        """Perform a 'forward pass' of the ray tracer thing.

        Arguments:
            X: The input image, must be the same shape as `input_shape`.

        Returns: The predicted labels as probabilities.

        Raises:
            AssertionError: if the shape of `X` does not match `input_shape`.
        """
        assert len(X.shape) - 1 == len(self.input_shape), \
            'Expected input to be a batch of samples, instead got a single ' \
            'sample.'
        assert X.shape[1:] == self.input_shape, "Expected input to be of the " \
                                                "shape %s, instead got %s." \
                                                % (self.input_shape, X.shape)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        output = [[None for _ in range(self.output_layer.n_cols)] for _ in range(self.output_layer.n_rows)]

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                pixel_output = X

                for w in self.W[row][col]:
                    pixel_output = pixel_output * w

                pixel_output = torch.sum(pixel_output, (1, 2))

                output[row][col] = pixel_output.reshape(-1, 1)

        output = torch.cat(tuple(output[0]), dim=1)

        return self.activation(output)

    def predict(self, X):
        """Perform a 'forward pass' of the ray tracer thing.

        Arguments:
            X: The input image, must be the same shape as `input_shape`.

        Returns: The predicted labels.

        Raises:
            AssertionError: if the shape of `X` does not match `input_shape`.
        """
        y = self.predict_proba(X)
        y = y.argmax(dim=1)

        return y

    def score(self, X, y):
        """Calculate the classification accuracy over the given samples.

        Arguments:
            X: The feature data.
            y: The labels.

        Returns: The classification accuracy as a ratio.
        """
        with torch.no_grad():
            y_pred = self.predict(X)

            if isinstance(y, np.ndarray):
                y = torch.tensor(y)

            return torch.mean((y_pred == y).double())

    def _broadcast_pixel_values(self):
        """Update the weights that share the same pixel so that they have the same value.

        Some elements in the weight matrices refer to the same pixel in the
        pixel grid, however the pixel value is represented as multiple entries
        in the weights matrices and are trained with back-propagation
        separately. In reality, a single pixel can only take on a single value,
        so this must be resolved by deciding on a single value for the pixel
        and assigning this value to each of the matrix elements that refer to
        the pixel.
        """
        for layer in range(self.n_hidden_layers):
            self.hidden_layers[layer].pixel_values = np.zeros(self.hidden_layer_shape)

        for row in range(self.output_layer.n_rows):
            for col in range(self.output_layer.n_cols):
                for layer in range(self.n_hidden_layers):

                    # The weights represent transparency values so they should be
                    # clamped to the interval [0.0, 1.0].
                    self.W[row][col][layer] = torch.clamp(self.W[row][col][layer], 0.0, 1.0)

                    # Adjust for pixels that are represented as separate values in the
                    # weight matrices but are actually a single entity.
                    for grid_coord in self.grid_W_map[row][col][layer]:
                        row_slice = self.grid_W_map[row][col][layer][grid_coord]['row_slice']
                        col_slice = self.grid_W_map[row][col][layer][grid_coord]['col_slice']

                        mean_pixel_value = self.W[row][col][layer][row_slice, col_slice].mean()
                        self.W[row][col][layer][row_slice, col_slice] = mean_pixel_value

                        self.hidden_layers[layer].pixel_values[grid_coord] += mean_pixel_value

        for layer in range(self.n_hidden_layers):
            self.hidden_layers[layer].pixel_values *= self.inverse_pixel_frequencies[layer]
            self.hidden_layers[layer].pixel_values = self.hidden_layers[layer].pixel_values.clip(1e-9, 1)

        self.W = self._get_W()
