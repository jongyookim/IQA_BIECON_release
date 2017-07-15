from __future__ import absolute_import, division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet.bn import batch_normalization


def linear(x):
    return x


class BatchNormLayer(object):
    """Batch normalization layer.
    Core algorithm is brought from Lasagne.
    (https://github.com/Lasagne/Lasagne)
    """
    layers = []

    def __init__(self, input_shape, layer_name=None, epsilon=1e-4, alpha=0.05):

        if len(input_shape) == 2:
            self.axes = (0,)
            shape = [input_shape[1]]
        elif len(input_shape) == 4:
            self.axes = (0, 2, 3)
            shape = [input_shape[0]]
        else:
            raise NotImplementedError

        self.layer_name = 'BN' if layer_name is None else layer_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.deterministic = False
        self.update_averages = True

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   name=layer_name + '_G', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=layer_name + '_B', borrow=True)

        self.mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=layer_name + '_mean', borrow=True)
        self.inv_std = theano.shared(
            np.ones(shape, dtype=theano.config.floatX),
            name=layer_name + '_inv_std', borrow=True)

        self.params = [self.gamma, self.beta]
        self.statistics = [self.mean, self.inv_std]
        BatchNormLayer.layers.append(self)

    def get_output(self, input, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))
        # input_inv_std = T.inv(T.sqrt(input.var(self.axes)) + 1E-6)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = self.deterministic
        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        update_averages = self.update_averages and not use_averages
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized

    def reset_mean_std(self):
        # reset mean and std
        self.mean.set_value(np.zeros(self.mean.get_value().shape,
                                     dtype=theano.config.floatX))
        self.inv_std.set_value(np.ones(self.std.get_value().shape,
                                       dtype=theano.config.floatX))

    @staticmethod
    def set_batch_norms_training(training):
        deterministic = False if training else True
        print(' - Batch norm layres: deterministic =', deterministic)
        for layer in BatchNormLayer.layers:
            layer.deterministic = deterministic
            layer.update_averages = not deterministic

    @staticmethod
    def reset_batch_norms_mean_std():
        print(' - Batch norm layres: reset mean and std')
        for layer in BatchNormLayer.layers:
            layer.reset_mean_std()


class BatchNormLayerTheano(object):
    """Batch normalization layer
    (Using theano.tensor.nnet.bn.batch_normalization)
    Core algorithm is brought from Lasagne.
    (https://github.com/Lasagne/Lasagne)
    """
    layers = []

    def __init__(self, input_shape, layer_name=None, activation=linear,
                 epsilon=1e-4, alpha=0.05):

        if len(input_shape) == 2:
            self.axes = (0,)
            shape = [input_shape[1]]
        elif len(input_shape) == 4:
            self.axes = (0, 2, 3)
            shape = [input_shape[0]]
        else:
            raise NotImplementedError

        self.layer_name = 'BN' if layer_name is None else layer_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.deterministic = False
        self.update_averages = True
        self.activation = activation
        self.act_name = activation.__name__

        self.gamma = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                   name=layer_name + '_G', borrow=True)
        self.beta = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=layer_name + '_B', borrow=True)

        self.mean = theano.shared(np.zeros(shape, dtype=theano.config.floatX),
                                  name=layer_name + '_mean', borrow=True)
        self.std = theano.shared(np.ones(shape, dtype=theano.config.floatX),
                                 name=layer_name + '_std', borrow=True)

        self.params = [self.gamma, self.beta]
        self.statistics = [self.mean, self.std]
        BatchNormLayer.layers.append(self)

    def get_output(self, input, **kwargs):
        input_mean = input.mean(self.axes)
        # input_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))
        input_std = T.sqrt(input.var(self.axes) + self.epsilon)

        # Decide whether to use the stored averages or mini-batch statistics
        use_averages = self.deterministic
        if use_averages:
            mean = self.mean
            std = self.std
        else:
            mean = input_mean
            std = input_std

        # Decide whether to update the stored averages
        update_averages = self.update_averages and not use_averages
        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_std = theano.clone(self.std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_std.default_update = ((1 - self.alpha) * running_std +
                                          self.alpha * input_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            std += 0 * running_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(list(range(input.ndim - len(self.axes))))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        std = std.dimshuffle(pattern)

        # normalize
        # normalized = (input - mean) * (gamma * std) + beta
        normalized = batch_normalization(
            input, gamma, beta, mean, std, mode='low_mem')
        return self.activation(normalized)

    def reset_mean_std(self):
        # reset mean and std
        self.mean.set_value(np.zeros(self.mean.get_value().shape,
                                     dtype=theano.config.floatX))
        self.std.set_value(np.ones(self.std.get_value().shape,
                                   dtype=theano.config.floatX))

    @staticmethod
    def set_batch_norms_training(training):
        deterministic = False if training else True
        print(' - Batch norm layres: deterministic =', deterministic)
        for layer in BatchNormLayer.layers:
            layer.deterministic = deterministic
            layer.update_averages = not deterministic

    @staticmethod
    def reset_batch_norms_mean_std():
        print(' - Batch norm layres: reset mean and std')
        for layer in BatchNormLayer.layers:
            layer.reset_mean_std()
