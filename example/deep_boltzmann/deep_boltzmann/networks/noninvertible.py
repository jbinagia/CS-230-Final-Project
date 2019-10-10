import keras
import tensorflow as tf
import numpy as np

from deep_boltzmann.networks import nonlinear_transform
from deep_boltzmann.networks import connect as _connect

class NormalTransformer(object):

    def __init__(self, mu_layers, sigma_layers):
        self.mu_layers = mu_layers
        self.sigma_layers = sigma_layers

    def _compute_x1(self, mu, log_sigma, w1):
        return mu + tf.exp(log_sigma) * w1

    def _compute_log_p1(self, mu, log_sigma, x1):
        return -tf.reduce_sum(log_sigma, axis=1) - 0.5 * tf.reduce_sum(((x1 - mu)/(tf.exp(log_sigma)))**2, axis=1)

    def connect(self, x0, w1):
        # evaluate mu and sigma
        mu = _connect(x0, self.mu_layers)
        log_sigma = _connect(x0, self.sigma_layers)
        # transform x
        #x1 = mu + sigma * w0
        self.x1 = keras.layers.Lambda(lambda args: self._compute_x1(args[0], args[1], args[2]))([mu, log_sigma, w1])
        # compute density
        #log_p1 = -tf.reduce_sum(sigma, axis=0) - 0.5 * tf.reduce_sum((self.x1 - mu)/sigma, axis=0)
        self.log_p1 = keras.layers.Lambda(lambda args: self._compute_log_p1(args[0], args[1], args[2]))([mu, log_sigma, self.x1])
        # return variable and density
        return self.x1, self.log_p1

class NormalResidualTransformer(object):

    def __init__(self, mu_layers, sigma_layers):
        self.mu_layers = mu_layers
        self.sigma_layers = sigma_layers

    def _compute_x1(self, x0, mu, log_sigma, w1):
        return x0 + mu + tf.exp(log_sigma) * w1

    def _compute_log_p1(self, x0, mu, log_sigma, x1):
        return -tf.reduce_sum(log_sigma, axis=1) - 0.5 * tf.reduce_sum(((x1 - x0 - mu)/(tf.exp(log_sigma)))**2, axis=1)

    def connect(self, x0, w1):
        # evaluate mu and sigma
        mu = _connect(x0, self.mu_layers)
        log_sigma = _connect(x0, self.sigma_layers)
        # transform x
        #x1 = mu + sigma * w0
        self.x1 = keras.layers.Lambda(lambda args: self._compute_x1(args[0], args[1], args[2], args[3]))([x0, mu, log_sigma, w1])
        # compute density
        #log_p1 = -tf.reduce_sum(sigma, axis=0) - 0.5 * tf.reduce_sum((self.x1 - mu)/sigma, axis=0)
        self.log_p1 = keras.layers.Lambda(lambda args: self._compute_log_p1(args[0], args[1], args[2], args[3]))([x0, mu, log_sigma, self.x1])
        # return variable and density
        return self.x1, self.log_p1

class NoninvNet(object):
    def __init__(self, dim, layers):
        self.dim = dim
        self.layers = layers
        self.log_p_total = None

    def connect(self):
        # x0 = 0
        self.x0 = keras.layers.Input(shape=(self.dim,))  # current noise input
        x_last = self.x0

        self.xs = []
        self.ws = []
        self.log_ps = []
        for layer in self.layers:
            # noise input
            w = keras.layers.Input(shape=(self.dim,))  # current noise input
            self.ws.append(w)
            # compute x and probability
            x, log_p = layer.connect(x_last, w)
            self.xs.append(x)  # new state
            self.log_ps.append(log_p)  # conditional generation probability
            # update x_last
            x_last = x
        # output
        self.x_out = self.xs[-1]
        # total probability
        self.log_p_total = keras.layers.Lambda(lambda arg: tf.reduce_sum(arg, axis=0))(self.log_ps)


    def log_probability(self):
        """ Computes the total log probability of the current sample"""
        return tf.reduce_sum(self.log_ps, axis=0)


def normal_transnet(dim, nlayers, mu_shape=(100, 100), mu_activation='relu',
                    sigma_shape=(100, 100), sigma_activation='tanh', residual=False,
                    **layer_args):
    """
    dim : int
        Dimension of variables
    nlayers : int
        Number of layers in the transformer
    mu_shape : int
        Number of hidden units in each nonlinear layer
    mu_activation : str
        Hidden-neuron activation functions used in the nonlinear layers
    sigma_shape : int
        Number of hidden units in each nonlinear layer
    sigma_activation : str
        Hidden-neuron activation functions used in the nonlinear layers

    """
    layers = []
    for l in range(nlayers):
        mu_net = nonlinear_transform(dim, nlayers=len(mu_shape)+1, nhidden=mu_shape,
                                     activation=mu_activation, **layer_args)
        sigma_net = nonlinear_transform(dim, nlayers=len(sigma_shape)+1, nhidden=sigma_shape,
                                        activation=sigma_activation, init_outputs=0, **layer_args)
        if residual:
            layer = NormalResidualTransformer(mu_net, sigma_net)
        else:
            layer = NormalTransformer(mu_net, sigma_net)
        layers.append(layer)
    ninvnet = NoninvNet(dim, layers)
    ninvnet.connect()
    return ninvnet
