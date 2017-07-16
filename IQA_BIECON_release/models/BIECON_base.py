from __future__ import absolute_import, division, print_function

import os
import numpy as np
import theano.tensor as T

from .model_basis import ModelBasis
from .model_record import Record
from ..layers import layers


class Model(ModelBasis):
    def __init__(self, model_config, rng=None):
        super(Model, self).__init__(model_config, rng)
        self.set_configs(model_config)

        self.layers['feat'] = []
        self.layers['feat_fc'] = []
        self.layers['reg_loc'] = []
        self.layers['reg_mos'] = []

        print('\nBIECON base model')
        print(' - Model file: %s' % (os.path.split(__file__)[1]))

        self.init_model()

    def set_configs(self, model_config):
        self.set_opt_configs(model_config)
        self.wl_loc = float(model_config.get('wl_loc', 1e2))
        self.wl_mos = float(model_config.get('wl_mos', 1e2))
        self.wr_l2 = float(model_config.get('wr_l2', 1e-4))
        self.dropout = model_config.get('use_dropout', False)
        self.update_wrt_loc = model_config.get(
            'update_wrt_loc', ['feat', 'feat_fc', 'reg_loc'])
        self.update_wrt_iqa = model_config.get(
            'update_wrt_iqa', ['feat', 'feat_fc', 'reg_mos'])

    def init_model(self):
        print(' - Feature conv layers')
        cur_key = 'feat'
        self.layers[cur_key] = []

        # Conv. layers
        self.layers[cur_key].append(layers.ConvLayer(
            input_shape=self.get_input_shape(),
            num_filts=64,
            filt_size=(5, 5),
            layer_name=cur_key + '/conv1',
            activation=layers.relu,
        ))

        self.layers[cur_key].append(layers.Pool2DLayer(
            input_shape=self.get_out_shape(cur_key),
            pool_size=(2, 2), mode='max'))

        self.layers[cur_key].append(layers.ConvLayer(
            input_shape=self.get_out_shape(cur_key),
            num_filts=64,
            filt_size=(5, 5),
            layer_name=cur_key + '/conv2',
            activation=layers.relu,
        ))

        self.layers[cur_key].append(layers.Pool2DLayer(
            input_shape=self.get_out_shape(cur_key),
            pool_size=(2, 2), mode='max'))

        # Reshaping layer
        self.layers[cur_key].append(
            layers.TensorToVectorLayer(self.get_out_shape(cur_key)))

        # Fully connected layers
        cur_key = 'feat_fc'
        self.layers[cur_key] = []

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape('feat'),
            n_out=1024,
            layer_name=cur_key + '/fc1',
            activation=layers.relu,
        ))

        if self.dropout:
            self.layers[cur_key].append(layers.DropoutLayer(p=0.5))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape(cur_key),
            n_out=512,
            layer_name=cur_key + '/fc2',
            activation=layers.relu,
        ))

        if self.dropout:
            self.layers[cur_key].append(layers.DropoutLayer(p=0.5))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape(cur_key),
            n_out=256,
            layer_name=cur_key + '/fc3',
            activation=layers.relu,
        ))

        if self.dropout:
            self.layers[cur_key].append(layers.DropoutLayer(p=0.5))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape(cur_key),
            n_out=128,
            layer_name=cur_key + '/fc4',
            activation=layers.relu,
        ))

        #######################################################################
        print(' - Regression metric layers')
        cur_key = 'reg_loc'
        self.layers[cur_key] = []

        if self.dropout:
            self.layers[cur_key].append(layers.DropoutLayer(p=0.5))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape('feat_fc'),
            n_out=128,
            layer_name=cur_key + '/fc1',
            activation=layers.relu,
        ))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape('feat_fc'),
            n_out=1,
            layer_name=cur_key + '/fc2',
            b_init=np.ones((1,), dtype='float32') * 0.5,
        ))

        #######################################################################
        print(' - Regression mos layers')
        cur_key = 'reg_mos'
        self.layers[cur_key] = []

        if self.dropout:
            self.layers[cur_key].append(layers.DropoutLayer(p=0.5))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape('feat_fc'),
            n_out=128,
            layer_name=cur_key + '/fc1',
            activation=layers.relu,
        ))

        self.layers[cur_key].append(layers.FCLayer(
            n_in=self.get_out_shape(cur_key),
            n_out=1,
            layer_name=cur_key + '/fc2',
            b_init=np.ones((1,), dtype='float32') * 0.5,
        ))

        #######################################################################

        super(Model, self).make_param_list()
        super(Model, self).show_num_params()

    def aggregation_fn(self, feat_vec):
        feat_avg = T.mean(feat_vec, axis=0, keepdims=True)
        return feat_avg
        # feat_std = T.std(feat_vec, axis=0, keepdims=True)
        # return T.concatenate([feat_avg, feat_std], axis=1)

    def feat_fn(self, x):
        out = self.get_key_layers_output(x, 'feat')
        return self.get_key_layers_output(out, 'feat_fc')

    def regress_loc_fn(self, feat_vec):
        return self.get_key_layers_output(feat_vec, 'reg_loc')

    def regress_mos_fn(self, feat_vec):
        return self.get_key_layers_output(feat_vec, 'reg_mos')

    def cost_reg_loc(self, x_c, met_s, n_img=None, bat2img_idx_set=None):
        """Get cost: regression onto local metroc scores
        """
        records = Record()
        # concatenate the image patches
        if bat2img_idx_set:
            # if dummy data with fixed size is given and current data is
            # overwritten on dummy data with size of n_patches,
            # pick current dataset with size of n_patches
            n_patches = bat2img_idx_set[n_img - 1][1]
            x_c_set = x_c[:n_patches]
            met_s_set = met_s[:n_patches]
        else:
            # if input is current data
            x_c_set = x_c
            met_s_set = met_s

        ######################################################################
        x_c_im = self.image_vec_to_tensor(x_c_set)
        met_s_im = self.image_vec_to_tensor(met_s_set)

        feat_vec = self.feat_fn(x_c_im)
        met_s_p = self.regress_loc_fn(feat_vec).flatten()

        met_s_mean = T.mean(met_s_set, axis=[1, 2, 3])
        loc_cost = self.get_cost_mse_mae(met_s_mean, met_s_p)

        # regularization
        l2_reg = self.get_l2_regularization(
            ['feat', 'feat_fc', 'reg_loc'], mode='sum')

        cost = self.add_all_losses_with_weight(
            [loc_cost, l2_reg],
            [self.wl_loc, self.wr_l2])

        # Parameters to record
        records.add_data('loc_mse', self.wl_loc * loc_cost)
        records.add_data('l2_reg', self.wr_l2 * l2_reg)

        # records.add_im_data('met_s_p', met_s_p_set)
        # records.add_im_data('met_s', met_s_set)

        records.add_imgs('x_c', x_c_im, caxis=[-0.25, 0.25])

        if bat2img_idx_set:
            def score_to_img(score, repeat=1):
                tmp = score.dimshuffle(0, 'x', 'x', 'x')
                tmp = T.extra_ops.repeat(tmp, repeat, axis=2)
                return T.extra_ops.repeat(tmp, repeat, axis=3)
            met_s_img = score_to_img(met_s_mean, 10)
            records.add_imgs('met_s', met_s_img, caxis='auto')
            met_s_p_img = score_to_img(met_s_p, 10)
            records.add_imgs('met_s_p', met_s_p_img, caxis='auto')

        return cost, records

    def cost_updates_reg_loc(self, x_c, met_s,
                             n_img=None, bat2img_idx_set=None):
        cost, records = self.cost_reg_loc(
            x_c, met_s, n_img=n_img, bat2img_idx_set=bat2img_idx_set)
        updates = self.get_updates_keys(cost, self.update_wrt_loc)
        return cost, updates, records

    def cost_nr_iqa(self, x_c, mos, n_img=None, bat2img_idx_set=None):
        records = Record()
        # concatenate the image patches
        if bat2img_idx_set:
            # if dummy data with fixed size is given and current data is
            # overwritten on dummy data with size of n_patches,
            # pick current dataset with size of n_patches
            n_patches = bat2img_idx_set[n_img - 1][1]
            x_c_set = x_c[:n_patches]
        else:
            # if input is current data
            x_c_set = x_c

        ######################################################################
        x_c_im = self.image_vec_to_tensor(x_c_set)
        # x_c_im = normalize_lowpass_subt(x_c_im, 3)

        feat_vec = self.feat_fn(x_c_im)

        # get feature vector and concatenate the mos_p set
        if bat2img_idx_set:
            # if patch based
            aggr_feat_list = []
            for idx in range(n_img):
                idx_from = bat2img_idx_set[idx][0]
                idx_to = bat2img_idx_set[idx][1]

                cur_feat_vec = feat_vec[idx_from: idx_to]
                cur_aggr_feat = self.aggregation_fn(cur_feat_vec)

                aggr_feat_list.append(cur_aggr_feat)

            aggr_feat = T.concatenate(aggr_feat_list, axis=0).flatten(2)
            # aggr_feat = T.stack(aggr_feat_list).flatten()
        else:
            # aggr_feat = self.regress_mos_fn(feat_vec).flatten()
            raise NotImplementedError

        ######################################################################
        # regress onto MOS
        mos_p = self.regress_mos_fn(aggr_feat).flatten()

        # MOS loss
        subj_loss = self.get_cost_mse_mae(mos, mos_p)

        # L2 regularization
        l2_reg = self.get_l2_regularization(
            ['feat', 'feat_fc', 'reg_mos'], mode='sum')

        cost = self.add_all_losses_with_weight(
            [subj_loss, l2_reg],
            [self.wl_mos, self.wr_l2])

        # Parameters to record
        records.add_data('subj', self.wl_mos * subj_loss)
        records.add_data('l2_reg', self.wr_l2 * l2_reg)

        records.add_im_data('mos_p', mos_p)
        records.add_im_data('mos_gt', mos)

        records.add_imgs('x_c', x_c_im, caxis=[-0.25, 0.25])

        return cost, records

    def cost_updates_nr_iqa(self, x_c, mos, n_img=None, bat2img_idx_set=None):
        cost, records = self.cost_nr_iqa(
            x_c, mos, n_img=n_img, bat2img_idx_set=bat2img_idx_set)
        updates = self.get_updates_keys(cost, self.update_wrt_iqa)
        return cost, updates, records

    def set_training_mode(self, training):
        # Decide behaviors of the model during training
        # Dropout
        self.set_dropout_on(training)
