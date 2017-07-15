from __future__ import absolute_import, division, print_function

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from .normalization import BatchNormLayer


# Activation functions
def linear(x):
    """linear: output = input"""
    return x


def tanh(x):
    """hyperbolic tangent"""
    return T.tanh(x)


def sigm(x):
    """sigmoid"""
    return T.nnet.sigmoid(x)


def relu(x, alpha=0.0):
    """
    Rectified linear unit
    return T.switch(x > 0, x, 0)
    """
    # return T.switch(x > 0, x, 0)
    return T.nnet.relu(x, alpha)


def lrelu(x, alpha=0.1):
    """
    Leaky relu
    return T.switch(x > 0, x, alpha * x)
    """
    # return T.switch(x > 0, x, alpha * x)
    return T.nnet.relu(x, alpha)


def elu(x, alpha=1.0):
    """
    Exponential LU
    return T.switch(x > 0, x, alpha * (T.exp(x) - 1))
    """
    # return T.switch(x > 0, x, alpha * (T.exp(x) - 1))
    return T.nnet.elu(x, alpha)


##############################################################################
class Layer(object):
    """
    Base class for layers
    """
    init_rng = np.random.RandomState(1235)
    # init_rng = np.random.RandomState()

    def __init__(self):
        self.params = []
        self.batch_norm = False
        self.rng = Layer.init_rng

    def has_batch_norm(self):
        return self.batch_norm

    def get_params(self):
        return self.params

    def get_output(self, input, **kwargs):
        raise NotImplementedError("get_output")

    def get_out_shape(self):
        return None

    def init_weight_he(self, shape, activation):
        # He et al. 2015
        if activation in [relu, elu]:
            gain = np.sqrt(2)
        elif activation == lrelu:
            alpha = 0.1
            gain = np.sqrt(2 / (1 + alpha ** 2))
        else:
            gain = 1.0

        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])

        std = gain * np.sqrt(1.0 / fan_in)
        W_values = np.asarray(
            self.rng.normal(0.0, std, size=shape),
            dtype=theano.config.floatX)
        return W_values

    def init_weight_xavier(self, shape, activation):
        # Xaiver
        if activation in [sigm]:
            gain = 4.0
        else:
            gain = 1.0

        if len(shape) == 2:
            fan_in, fan_out = shape
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])
            fan_out = shape[0] * np.prod(shape[2:])

        W_bound = gain * np.sqrt(6. / (fan_in + fan_out))
        W_values = np.asarray(
            self.rng.uniform(low=-W_bound, high=W_bound, size=shape),
            dtype=theano.config.floatX)
        return W_values


##############################################################################
# Layers
class FCLayer(Layer):
    """
    Fully connected layer.
    Parameters
    ----------
        n_in: input feature dimension
        n_out: output feature dimension

    Attributes
    ----------
        W: Theano shared variable representing the filter weights.
        b: Theano shared variable representing the biases.
    """
    def __init__(self, n_in, n_out, W=None, b=None, b_init=None,
                 layer_name=None, activation=linear, batch_norm=False):
        super(FCLayer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.act_name = activation.__name__
        self.batch_norm = batch_norm
        self.layer_name = 'FC' if layer_name is None else layer_name

        self.params = []
        if W is None:
            W_values = self.init_weight_he((n_in, n_out), self.activation)
            self.W = theano.shared(value=W_values, name=self.layer_name + '_W',
                                   borrow=True)
            self.params += [self.W]
        else:
            self.W = W

        if b is None:
            if b_init is not None:
                b_values = b_init
            else:
                b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name=self.layer_name + '_b',
                                   borrow=True)
            self.params += [self.b]
        else:
            self.b = b

        # Batch normalization
        if self.batch_norm:
            self.bn_layer = BatchNormLayer(input_shape=(self.n_in, self.n_out),
                                           layer_name='BN_' + layer_name)
            self.b = None
            self.params = [self.W] + self.bn_layer.params

        # Show information
        print('  # %s (FC): in = %d -> out = %d,' % (
            self.layer_name, self.n_in, self.n_out), end=' ')
        print('act.: %s,' % self.act_name, end=' ')
        if self.batch_norm:
            print('Batch norm.')
        else:
            print('')

    def get_output(self, input, **kwargs):
        lin_output = T.dot(input, self.W)

        if self.batch_norm:
            lin_output = self.bn_layer.get_output(lin_output)
        else:
            lin_output += self.b

        return self.activation(lin_output)

    def get_out_shape(self):
        return self.n_out


class ConvLayer(Layer):
    """
    Convolutional layer.
    Parameters
    ----------
        filter_shape: (number of filters, num input feature maps,
                       filter height, filter width)
                      or (number of filters, filter size)
        input_shape: (batch size, num input feature maps,
                      image height, image width)
    Attributes
    ----------
        W : Theano shared variable representing the filter weights.
        b : Theano shared variable representing the biases.
    """
    def __init__(self, input_shape, filter_shape=None,
                 num_filts=None, filt_size=None,
                 W=None, b=None, W_init=None, b_init=None,
                 mode='half', subsample=(1, 1),
                 layer_name=None, activation=linear, batch_norm=False,
                 show_info=True):
        super(ConvLayer, self).__init__()

        # Make filter shape
        if filter_shape is None:
            assert num_filts is not None and filt_size is not None
            if isinstance(filt_size, (list, tuple)):
                assert len(filt_size) == 2
                filter_shape = (num_filts, input_shape[1]) + filt_size
            else:
                filter_shape = (num_filts, input_shape[1],
                                filt_size, filt_size)
        else:
            assert input_shape[1] == filter_shape[1]

        # Calculate output shape and validate
        if isinstance(mode, tuple):
            self.mode = mode
            self.out_size = [
                input_shape[i] - filter_shape[i] + 2 * self.mode[i - 2] + 1
                for i in range(2, len(input_shape))]
        else:
            self.mode = mode.lower()
            if self.mode == 'valid':
                self.out_size = [input_shape[i] - filter_shape[i] + 1
                                 for i in range(2, len(input_shape))]
            elif self.mode == 'half':
                self.out_size = input_shape[2:]
            elif self.mode == 'full':
                self.out_size = [input_shape[i] - filter_shape[i] - 1
                                 for i in range(2, len(input_shape))]
            else:
                raise ValueError('Invalid mode: %s' % self.mode)
        self.out_size = tuple(self.out_size)
        for sz in self.out_size:
            if sz < 1:
                raise ValueError('Invalid feature size: (%s).' %
                                 ', '.join([str(i) for i in self.out_size]))

        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.activation = activation
        self.act_name = activation.__name__
        self.batch_norm = batch_norm
        self.layer_name = 'Conv' if layer_name is None else layer_name
        self.subsample = subsample

        # Initialize parameters
        self.params = []
        if W is None:
            if W_init is not None:
                W_values = W_init
            else:
                W_values = self.init_weight_he(filter_shape, self.activation)
            self.W = theano.shared(W_values, name=self.layer_name + '_W',
                                   borrow=True)
            self.params += [self.W]
        else:
            self.W = W

        self.no_bias = False
        if b is None:
            if b_init is not None:
                b_values = b_init
            else:
                b_values = np.zeros((filter_shape[0],),
                                    dtype=theano.config.floatX)
            self.b = theano.shared(b_values, name=self.layer_name + '_b',
                                   borrow=True)
            self.params += [self.b]
        elif b is False:
            self.b = 0.0
            self.no_bias = True
        else:
            self.b = b

        # Batch normalization
        if batch_norm:
            self.bn_layer = BatchNormLayer(input_shape=filter_shape,
                                           layer_name='BN_' + self.layer_name)
            self.b = None
            self.params = [self.W] + self.bn_layer.params

        # Show information
        if show_info:
            print('  # %s (Conv-%s):' % (layer_name, mode), end=' ')
            print('flt.(%s),' % ', '.join(
                [str(i) for i in self.filter_shape]), end=' ')
            print('in.(%s),' % ', '.join(
                [str(i) for i in self.input_shape[1:]]), end=' ')
            print('act.: %s,' % self.act_name, end=' ')
            if self.batch_norm:
                print('Batch norm.')
            else:
                print('')
            if self.subsample != (1, 1):
                print('    subsample (%s) -> (%s)' % (
                    ', '.join([str(i) for i in self.input_shape[1:]]),
                    ', '.join([str(i) for i in self.get_out_shape()[1:]])))

    def get_output(self, input, **kwargs):
        var_shape = kwargs.get('var_shape', False)
        if var_shape:
            input_shape = None
        else:
            input_shape = self.input_shape
        lin_output = conv2d(
            input=input,
            filters=self.W,
            filter_shape=self.filter_shape,
            border_mode=self.mode,
            subsample=self.subsample,
            input_shape=input_shape
        )

        if self.batch_norm:
            lin_output = self.bn_layer.get_output(lin_output)
        elif not self.no_bias:
            lin_output += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.activation(lin_output)

    def get_out_shape(self, after_ss=True):
        out_size = self.out_size
        if after_ss:
            out_size = [(out_size[i] + self.subsample[i] - 1) //
                        self.subsample[i] for i in range(len(out_size))]

        return (self.input_shape[0], self.filter_shape[0]) + tuple(out_size)


class TensorToVectorLayer(Layer):
    def __init__(self, input_shape):
        super(TensorToVectorLayer, self).__init__()
        self.input_shape = input_shape
        print('  # tensor to vector: (%s) -> %d' % (
            ', '.join([str(i) for i in self.input_shape[1:]]),
            np.prod(self.input_shape[1:])))

    def get_output(self, input, **kwargs):
        return input.flatten(2)

    def get_out_shape(self):
        return np.prod(self.input_shape[1:])


##############################################################################
# Dropout
class DropoutLayer(Layer):
    layers = []

    def __init__(self, p=0.5, rescale=True):
        super(DropoutLayer, self).__init__()

        self._srng = RandomStreams(self.rng.randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.deterministic = False
        DropoutLayer.layers.append(self)
        print('  # Dropout: p = %.2f' % (self.p))

    def get_output(self, input, **kwargs):
        if self.deterministic or self.p == 0:
            return input

        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)
            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            return input * self._srng.binomial(input.shape, p=retain_prob,
                                               dtype=input.dtype)

    @staticmethod
    def set_dropout_training(training):
        deterministic = False if training else True
        # print(' - Dropout layres: deterministic =', deterministic)
        for layer in DropoutLayer.layers:
            layer.deterministic = deterministic


class Pool2DLayer(Layer):
    """
    Downscale the input by a specified factor.

    Parameters
    ----------
        padding
            (tuple of two integers) - (pad_h, pad_w),
            pad zeros to extend beyond four borders of the images,
            pad_h is the size of the top and bottom margins,
            and pad_w is the size of the left and right margins.
        ignore_border:
            (bool (default None, will print a warning and set to False))
            When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
            (3,3) otherwise.
        mode:
            ({'max', 'sum', 'average_inc_pad', 'average_exc_pad'})
    """
    def __init__(self, input_shape, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DLayer, self).__init__()

        self.input_shape = input_shape
        self.pool_size = pool_size

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = stride

        self.pad = pad

        self.ignore_border = ignore_border
        self.mode = mode
        print('  # Pool-%s (%s) -> (%s)' % (
            mode,
            ', '.join([str(i) for i in self.input_shape[1:]]),
            ', '.join([str(i) for i in self.get_out_shape()[1:]])))

    def get_output(self, input, **kwargs):
        pooled = pool_2d(input,
                         ws=self.pool_size,
                         stride=self.stride,
                         ignore_border=self.ignore_border,
                         pad=self.pad,
                         mode=self.mode,
                         )
        return pooled

    def get_out_shape(self):
        output_shape = list(self.input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(self.input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border)

        output_shape[3] = pool_output_length(self.input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border)

        return tuple(output_shape)


def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length
