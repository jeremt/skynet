
from math import tanh
from random import random

class Connection(object):

    def __init__(self):
        self.weight = random() * 2. - 1.
        self.delta = 0.0

    def update(self, eta, alpha, delta):
        self.weight += eta * delta + alpha * self.delta
        self.delta = delta

    def __repr__(self):
        return "Connection(weight=%s, delta=%s)" % (self.weight, self.delta)

class SkyNet(object):

    def __init__(
        self, nb_input=1, nb_hidden=1, nb_output=1,
        activation_fn=None, activation_derivative=None,
        eta=0.2, alpha=0.02,
        ):
        """ Create a neural network with three layers (input, hidden
            and output). You can also provide a custom activation function,
            a custom momentum or learning rate.
        """
        
        self.activation_fn = activation_fn or tanh
        self.activation_derivative = activation_derivative or (lambda x: 1-x*x)

        self.eta = eta
        self.alpha = alpha

        # Number of input, hidden and output nodes.
        self.nb_input = nb_input + 1 # +1 for bias node
        self.nb_hidden = nb_hidden + 1 # +1 for bias node
        self.nb_output = nb_output

        # Create neurons layers.
        self.input_neurons = [1.0] * self.nb_input
        self.hidden_neurons = [1.0] * self.nb_hidden
        self.output_neurons = [1.0] * self.nb_output

        # Init neurons' gradients to 0.
        self.output_gradients = [0.0] * self.nb_output
        self.hidden_gradients = [0.0] * self.nb_hidden

        def create_matrix(x, y):
            return [[Connection() for j in range(y)] for i in range(x)]

        self.input_weights = create_matrix(self.nb_input, self.nb_hidden)
        self.hidden_weights = create_matrix(self.nb_hidden, self.nb_output)

    def feed_forward(self, inputs):
        """ Feeds the network with the given `inputs` from the input layer
            to the output layer of neurons.
        """
        if len(inputs) != self.nb_input - 1:
            raise ValueError("Inputs size should be %d." % self.nb_input - 1)

        # input activations
        for i in range(self.nb_input - 1):
            self.input_neurons[i] = inputs[i]

        # hidden activations
        for j in range(self.nb_hidden - 1):
            total = 0.0
            for i in range(self.nb_input):
                total += self.input_neurons[i] * self.input_weights[i][j].weight
            self.hidden_neurons[j] = self.activation_fn(total)

        # output activations
        for k in range(self.nb_output):
            total = 0.0
            for j in range(self.nb_hidden):
                total += self.hidden_neurons[j] * self.hidden_weights[j][k].weight
            self.output_neurons[k] = self.activation_fn(total)

        return self.output_neurons

    def back_propagate(self, targets):
        """ Update weights according targets. """

        if len(targets) != self.nb_output:
            raise ValueError("Targets size should be %d." % self.nb_output)

        # Compute output gradients.
        for i in range(self.nb_output):
            self.output_gradients[i] = targets[i] - self.output_neurons[i]
            self.output_gradients[i] *= self.activation_derivative(self.output_neurons[i])

        # Compute hidden gradients.
        for i in range(self.nb_hidden):
            error = 0.0
            for j in range(self.nb_output):
                error += self.output_gradients[j] * \
                    self.hidden_weights[i][j].weight
            self.hidden_gradients[i] = \
                self.activation_derivative(self.hidden_neurons[i]) * error

        # Update hidden from output.
        for i in range(self.nb_hidden):
            for j in range(self.nb_output):
                delta = self.output_gradients[j] * self.hidden_neurons[i]
                self.hidden_weights[i][j].update(self.eta, self.alpha, delta)

        # Update input from hidden.
        for i in range(self.nb_input):
            for j in range(self.nb_hidden):
                delta = self.hidden_gradients[j] * self.input_neurons[i]
                self.input_weights[i][j].update(self.eta, self.alpha, delta)

        # Compute error between target and output neurons.
        error = 0.0
        for k in range(len(targets)):
            delta = targets[k] - self.output_neurons[k]
            error += 0.5 * delta * delta
        return error
