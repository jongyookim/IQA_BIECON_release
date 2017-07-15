from __future__ import absolute_import, division, print_function

import pickle
from collections import OrderedDict

import numpy as np
import theano.tensor as T
from functools import reduce

from .. import optimizer
from ..layers import layers


class ModelBasis(object):
    """
    Arguments
        model_config: model configuration dictionary

    Attributes of model_config
        input_size: input image size, (height, width).
        num_ch: number of input channels
        mae: mean absolute error (MAE) / mean square error (MSE) ratio.
             0: only MSE / 1: only MAE
        lr: initial learning rate
    """

    def __init__(self, model_config={}, rng=None):
        # Check input_size
        input_size = model_config.get('input_size', None)
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        elif isinstance(input_size, (list, tuple)):
            assert len(input_size) == 2
            self.input_size = tuple(input_size)
        else:
            raise ValueError('Wrong input_size:', input_size)

        self.num_ch = model_config.get('num_ch', None)
        assert self.num_ch is not None

        self.input_shape = (None, self.num_ch) + self.input_size

        self.mae = float(model_config.get('mae', 0.0))
        self.opt = optimizer.Optimizer()
        self.set_opt_configs(model_config)

        if rng is None:
            rng = np.random.RandomState(1234)
        self.rng = rng

        self.layers = OrderedDict()
        self.params = OrderedDict()

    def set_opt_configs(self, model_config):
        self.lr = float(model_config.get('lr', 1e-3))
        self.opt_scheme = model_config.get('opt_scheme', 'adam')
        self.opt.set_learning_rate(self.lr)

    ###########################################################################
    # Functions for cost calculation

    def get_l2_regularization(self, layer_keys=None, mode='sum'):
        if layer_keys is None:
            layer_keys = list(self.layers.keys())
        l2 = []
        if mode == 'sum':
            for key in layer_keys:
                for layer in self.layers[key]:
                    if hasattr(layer, 'W'):
                        l2.append(T.sum(layer.W ** 2))
            return T.sum(l2)
        elif mode == 'mean':
            for key in layer_keys:
                for layer in self.layers[key]:
                    if hasattr(layer, 'W'):
                        l2.append(T.mean(layer.W ** 2))
            return T.mean(l2)
        else:
            raise NotImplementedError

    def get_cost_mse_mae(self, x, y):
        diff = x - y
        if self.mae == 0:
            return T.mean(diff ** 2)
        elif self.mae > 0 and self.mae < 1.0:
            return ((1.0 - self.mae) * T.mean(diff ** 2) +
                    self.mae * T.mean(T.abs_(diff)))
        else:
            return T.mean(T.abs_(diff))

    def get_mse(self, x, y, return_map=False):
        if return_map:
            return (x - y) ** 2
        else:
            return T.mean(((x - y) ** 2).flatten(2), axis=1)

    def add_all_losses_with_weight(self, losses, weights):
        """Add the losses with the weights multiplied.
        If the weight is 0, the corresponding loss is ignored.
        """
        assert len(losses) == len(weights)
        loss_list = []
        for loss, weight in zip(losses, weights):
            if weight != 0:
                loss_list.append(weight * loss)
        return reduce(lambda x, y: x + y, loss_list)

    ###########################################################################
    # Functions to help build layers

    def get_input_shape(self, batch_size=None):
        """Get the input shape of the Mode.
        Returns
        -------
            (batch_size, self.num_ch) + self.input_size
        """
        return (batch_size, self.num_ch) + self.input_size

    def get_out_shape(self, key, nth=-1):
        """Get the `nth` output shape in the `key` layers
        """
        if nth < 0:
            idx = len(self.layers[key]) + nth
        else:
            idx = nth
        out_sh = None
        while out_sh is None:
            if idx < 0:
                raise ValueError('Cannot obtain the output size from %s' % key)
            out_sh = self.layers[key][idx].get_out_shape()
            idx = idx - 1
        return out_sh

    def get_conc_shape(self, key0, key1):
        """Get the concatenated shape of the ouputs of
        `key0` and `key1` layers
        """
        prev_sh0 = self.get_out_shape(key0)
        prev_sh1 = self.get_out_shape(key1)
        if isinstance(prev_sh0, (list, tuple)):
            assert prev_sh0[0] == prev_sh1[0]
            assert prev_sh0[2:] == prev_sh1[2:]
            return (prev_sh0[0], prev_sh0[1] + prev_sh1[1]) + prev_sh0[2:]
        else:
            return prev_sh0 + prev_sh1

    ###########################################################################
    # Functions to help make computation graph

    def image_vec_to_tensor(self, input):
        """Reshape input into 4D tensor.
        """
        # im_sh = (-1, self.input_size[0],
        #          self.input_size[1], self.num_ch)
        # return input.reshape(im_sh).dimshuffle(0, 3, 1, 2)
        return input.dimshuffle(0, 3, 1, 2)

    # def tensor_to_image_vec(self, input):
    #     """Reshape 4D tensor into input."""
    #     return input.dimshuffle(0, 2, 3, 1).flatten(2)

    def get_key_layers_output(self, input, key, var_shape=False):
        """Put input to the `key` layers and get output.
        """
        prev_out = input
        for layer in self.layers[key]:
            prev_out = layer.get_output(prev_out, var_shape=var_shape)
        return prev_out

    def get_updates(self, cost, wrt_params):
        return self.opt.get_updates_cost(cost, wrt_params, self.opt_scheme)

    def get_updates_keys(self, cost, keys=[], params=[]):
        wrt_params = []
        for key in keys:
            wrt_params += self.params[key]
        if params:
            wrt_params += params

        print(' - Update w.r.t.: %s' % ', '.join(keys))
        return self.opt.get_updates_cost(cost, wrt_params, self.opt_scheme)

    ###########################################################################
    # Functions to contol batch normalization and dropout layers

    def get_batch_norm_layers(self, keys=[]):
        layers = []
        for key in list(self.layers.keys()):
            layers += self.bn_layers[key]
        return layers

    def set_batch_norm_update_averages(self, update_averages, keys=[]):
        # if update_averages:
        #     print(' - Batch norm: update the stored averages')
        # else:
        #     print(' - Batch norm: not update the stored averages')
        layers = self.get_batch_norm_layers(keys)
        for layer in layers:
            layer.update_averages = update_averages

    def set_batch_norm_training(self, training, keys=[]):
        # if training:
        #     print(' - Batch norm: use mini-batch statistics')
        # else:
        #     print(' - Batch norm: use the stored statistics')
        layers = self.get_batch_norm_layers(keys)
        for layer in layers:
            layer.deterministic = not training

    def set_dropout_on(self, training):
        layers.DropoutLayer.set_dropout_training(training)

    def set_training_mode(self, training):
        """Decide the behavior of batch normalization and dropout.
        Parameters
        ----------
            training: boolean
                True: trainig mode / False: testing mode.
        """

        # Decide behaviors of the model during training
        # Batch normalization
        l_keys = [key for key in list(self.layers.keys())]
        self.set_batch_norm_update_averages(training, l_keys)
        self.set_batch_norm_training(training, l_keys)

        # Dropout
        self.set_dropout_on(training)

    ###########################################################################
    # Functions to help deal with parameters of the model

    def make_param_list(self):
        """collect all the parameters from `self.layers` and
        store into `self.params['layer_key']`
        """
        self.params, self.bn_layers = {}, {}

        for key in list(self.layers.keys()):
            self.params[key] = []
            self.bn_layers[key] = []
            for layer in self.layers[key]:
                if layer.get_params():
                    self.params[key] += layer.get_params()
                if layer.has_batch_norm():
                    self.bn_layers[key].append(layer.bn_layer)

    def show_num_params(self):
        """Dislay the number of paraemeters for each layer_key.
        """
        paramscnt = {}
        for key in list(self.layers.keys()):
            paramscnt[key] = 0
            for p in self.params[key]:
                paramscnt[key] += np.prod(p.get_value(borrow=True).shape)
            if paramscnt[key] > 0:
                print(' - Num params %s:' % key, '{:,}'.format(paramscnt[key]))

    def get_params(self, layer_keys=None):
        """Get concatenated parameter list
        from layers belong to layer_keys"""
        if layer_keys is None:
            layer_keys = list(self.layers.keys())

        params = []
        bn_mean_std = []
        for key in layer_keys:
            params += self.params[key]

        for key in layer_keys:
            for layer in self.bn_layers[key]:
                bn_mean_std += layer.statistics
        params += bn_mean_std
        return params

    def save(self, filename):
        """Save parameters to file.
        """
        params = self.get_params()
        with open(filename, 'wb') as f:
            pickle.dump(params, f, protocol=2)
            # pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(' = Save params: %s' % (filename))

    def load(self, filename):
        """Load parameters from file.
        """
        params = self.get_params()
        with open(filename, 'rb') as f:
            newparams = pickle.load(f)

        assert len(newparams) == len(params)
        for p, new_p in zip(params, newparams):
            if p.name != new_p.name:
                print((' @ WARNING: Different name - (loaded) %s != %s'
                      % (new_p.name, p.name)))
            new_p_sh = new_p.get_value(borrow=True).shape
            p_sh = p.get_value(borrow=True).shape
            if p_sh != new_p_sh:
                # print(new_p.name, p_sh, new_p_sh)
                print(' @ WARNING: Different shape %s - (loaded)' % new_p.name,
                      new_p_sh, end='')
                print(' !=', p_sh)
                continue
            p.set_value(new_p.get_value())
        print(' = Load all params: %s ' % (filename))

    def load_params_keys(self, layer_keys, filename):
        """Load the selecte parameters from file.
        Parameters from layers belong to layer_keys.
        """
        print(' = Load params: %s (keys = %s)' % (
            filename, ', '.join(layer_keys)))
        to_params = self.get_params(layer_keys)
        with open(filename, 'rb') as f:
            from_params = pickle.load(f)

        # Copy the params having same shape and name
        copied_idx = []
        for fidx, f_param in enumerate(from_params):
            f_val = f_param.get_value(borrow=True)
            for tidx, t_param in enumerate(to_params):
                t_val = t_param.get_value(borrow=True)
                if f_val.shape == t_val.shape and f_param.name == t_param.name:
                    t_param.set_value(f_val)
                    del to_params[tidx]
                    copied_idx.append(fidx)
                    break
        # print(' = Copied from_param: ', [
        #     from_params[idx] for idx in copied_idx])
        if to_params:
            print(' = Not existing to_param: ', to_params)
