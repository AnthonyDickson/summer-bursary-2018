import numpy as np
from sklearn.model_selection import train_test_split
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
            x: the input.

        Returns: the input transformed with the softmax function.
        """
        z = torch.exp(x)

        return z / torch.sum(z, axis=0)

    @staticmethod
    def sigmoid(x, alpha=0):
        z = torch.exp(x - alpha)
        return z / (z + 1)


class Losses:
    """Implements a couple of loss functions."""

    @staticmethod
    def log_loss(true_label, predicted_prob):
        if true_label == 1:
            return -torch.log(predicted_prob)
        else:
            return -torch.log(1 - predicted_prob)


class RayTracerThing:
    """This thing does some stuff."""

    def __init__(self, input_shape, n_layers=0,
                 hidden_layer_shape=None, activation_func=Activations.identity,
                 learning_rate=0.01):
        """Create a ray tracer thing (need to think of a better name).

        Arguments:
            input_shape: The shape of the input image.
            n_layers: The number of 'hidden' layers to add.
            hidden_layer_shape: The shape of the 'hidden' layers. If set to
                                None, `hidden_layer_shape` defaults to
                                `input_shape`.
            learning_rate: The learning rate.
        """
        if hidden_layer_shape is None:
            hidden_layer_shape = input_shape

        self.input_shape = input_shape
        self.layer_shape = hidden_layer_shape
        self.output_shape = (1, 1)
        self.n_layers = n_layers
        self.activation = activation_func
        self.learning_rate = learning_rate

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

    def fit(self, X, y, val_data=0.2, n_epochs=100, batch_size='none', early_stopping=20):
        """Fit the classifier.

        Arguments:
            X: The feature data.
            y: The label data.
            val_data: The data that should be used for validation. This can
                      be a tuple containing the validation X and y, or a ratio
                      or an integer indicating how many instances from `X` and
                      `y` should be used for validation.
            n_epochs: How many epochs to fit the model for.
            batch_size: Not implemented yet.
            early_stopping: How many epochs to stop after if the loss has not
                            improved (i.e. decreased).
        """
        assert len(X) == len(y), 'Inputs X and y should be the same length.'

        try:
            # assume val_data is a tuple
            X_val, y_val = val_data
        except TypeError:  # val_data was not a tuple.
            X, X_val, y, y_val = train_test_split(X, y, test_size=val_data)

        # early stopping stuff
        best_loss = float('inf')
        n_epochs_no_improvement = 0
        patience = early_stopping

        n_train = len(X)
        n_val = len(X_val)

        for epoch in range(n_epochs):
            gradients = torch.zeros(n_train, self.n_layers, *self.layer_shape)
            train_loss = 0
            train_accuracy = 0

            image_i = 0

            self.zero_grad()

            for image, label in zip(X, y):
                out = self.predict_proba(image)
                loss = Losses.log_loss(label, out)

                loss.backward()

                for layer in range(self.n_layers):
                    gradients[image_i][layer] = self.W[layer].grad.clone()

                train_loss += loss

                predicted = 0 if out < 0.5 else 1

                if predicted == label:
                    train_accuracy += 1

                image_i += 1

            train_loss = train_loss / n_train
            train_accuracy = train_accuracy / n_train

            val_loss = 0
            val_accuracy = 0

            with torch.no_grad():
                mean_grad = gradients.mean(dim=0)

                for layer in range(self.n_layers):
                    self.W[layer] = self.W[layer] - self.learning_rate * mean_grad[layer]

                self.broadcast_pixel_values()

                for image, label in zip(X_val, y_val):
                    out = self.predict_proba(image)
                    loss = Losses.log_loss(label, out)
                    predicted = 0 if out < 0.5 else 1

                    if predicted == label:
                        val_accuracy += 1
                    val_loss += loss

            val_loss = val_loss / n_val
            val_accuracy = val_accuracy / n_val

            print('Epoch %d of %d - train_loss: %.4f - train_acc: %.4f - '
                  'val_loss: %.4f - val_acc: %.4f'
                  % (epoch + 1, n_epochs,
                     train_loss, train_accuracy,
                     val_loss, val_accuracy),
                  end='\r')

            if val_loss < best_loss:
                best_loss = val_loss
                n_epochs_no_improvement = 0
            else:
                n_epochs_no_improvement += 1

            if n_epochs_no_improvement > patience:
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
        assert X.shape == self.input_shape, "Expected input to be of the " \
                                            "shape %s, instead got %s." \
                                            % (self.input_shape, X.shape)

        output = torch.tensor(X)

        for w in self.W:
            output = output * w

        output = torch.sum(output)  # (1, 2))

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

        return 0 if y < 0.5 else 1

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
