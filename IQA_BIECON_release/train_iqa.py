from __future__ import absolute_import, division, print_function

import os
from importlib import import_module

import numpy as np
import theano
import theano.tensor as T

from .config_parser import config_parser, dump_config
from .data_load.data_loader_IQA import DataLoader
from .trainer import Trainer


def train_biecon(config_file, section, snap_path,
                 output_path=None, snap_file=None, loc_snap_file=None,
                 tr_te_file=None, epoch_loc=0, epoch_nr=0):
    """
    Train BIECON model with two stages.
    """
    db_config, model_config, train_config = config_parser(
        config_file, section)

    # Check snapshot file
    if snap_file is not None:
        assert os.path.isfile(snap_file), \
            'Not existing snap_file: %s' % snap_file

    # Load data
    data_loader = DataLoader(db_config)
    train_data, test_data = data_loader.load_data_tr_te(tr_te_file)

    # Create model
    opt_scheme_list, lr_list = make_opt_config_list(model_config)
    model = create_model(
        model_config, train_data.patch_size, train_data.num_ch)
    if snap_file is not None:
        model.load(snap_file)
    if loc_snap_file is not None:
        model.load_params_keys(['feat', 'feat_fc', 'reg_loc'], loc_snap_file)

    # Create trainer
    trainer = Trainer(train_config, snap_path, output_path)

    # Store current configuration file
    dump_config(os.path.join(snap_path, 'config.yaml'),
                db_config, model_config, train_config)

    ###########################################################################
    # Train the model
    prefix = train_config.get('prefix', '')

    x_c = T.ftensor4('x_c')
    met_s = T.ftensor4('met_s')
    mos_set = T.vector('mos_set')
    bat2img_idx_set = T.imatrix('bat2img_idx_set')

    # Train local metric scores
    if epoch_loc > 0:
        prefix2 = 'LOC_'
        batch_size_loc = int(train_config.get('batch_size_loc', 512))
        model.set_opt_configs(opt_scheme=opt_scheme_list[0], lr=lr_list[0])

        score = run_reg_loc_pw(
            train_data, test_data, model, trainer, epoch_loc, batch_size_loc,
            x_c=x_c, met_s=met_s, prefix2=prefix2)

        # Show information after train
        print("Best score0: {:.3f}, score1: {:.3f}, epoch: {:d}".format(
            score[0], score[1], score[2]))
        model.load(os.path.join(
            snap_path, prefix + prefix2 + 'snapshot_best.npy'))

    # Train NR-IQA
    if epoch_nr > 0:
        prefix2 = 'NR_'
        batch_size_nr = int(train_config.get('batch_size_nr', 8))
        model.set_opt_configs(opt_scheme=opt_scheme_list[1], lr=lr_list[1)

        score = run_nr_iqa(
            train_data, test_data, model, trainer, epoch_nr, batch_size_nr,
            x_c=x_c, mos_set=mos_set, bat2img_idx_set=bat2img_idx_set,
            prefix2=prefix2)

        # Show information after train
        print("Best SRCC: {:.3f}, PLCC: {:.3f}, epoch: {:d}".format(
            score[0], score[1], score[2]))


def run_reg_loc_pw(train_data, test_data, model, trainer, epochs, batch_size,
                   x_c=None, met_s=None, prefix2='loc_'):
    """
    @type model: .models.model_basis.ModelBasis
    @type train_data: .data_load.dataset.Dataset
    @type test_data: .data_load.dataset.Dataset
    """
    # Make dummy shared dataset
    sh = model.get_input_shape()
    np_set_d = np.zeros((batch_size, sh[2], sh[3], sh[1]), dtype='float32')
    shared_set_d = theano.shared(np_set_d, borrow=True)
    sh = train_data.loc_size
    np_met_s = np.zeros((batch_size, sh[0], sh[1], 1), dtype='float32')
    shared_met_s = theano.shared(np_met_s, borrow=True)

    train_data.set_patchwise()
    test_data.set_patchwise()

    print('\nCompile theano function: Regress on local metric scores', end='')
    print(' (patchwise / low GPU memory)')
    if x_c is None:
        x_c = T.ftensor4('x_c')
    if met_s is None:
        met_s = T.ftensor4('met_s')

    print(' (Make training model)')
    model.set_training_mode(True)
    cost, updates, rec_train = model.cost_updates_reg_loc(
        x_c, met_s)
    outputs = [cost] + rec_train.get_function_outputs(train=True)

    train_model = theano.function(
        [],
        [output for output in outputs],
        updates=updates,
        givens={
            x_c: shared_set_d,
            met_s: shared_met_s
        },
        on_unused_input='warn'
    )

    print(' (Make testing model)')
    model.set_training_mode(False)
    cost, rec_test = model.cost_reg_loc(x_c, met_s)
    outputs = [cost] + rec_test.get_function_outputs(train=False)

    test_model = theano.function(
        [],
        [output for output in outputs],
        givens={
            x_c: shared_set_d,
            met_s: shared_met_s
        },
        on_unused_input='warn'
    )

    def get_train_outputs():
        res = train_data.next_batch(batch_size)
        shared_set_d.set_value(res['dis_data'])
        shared_met_s.set_value(res['loc_data'])
        return train_model()

    def get_test_outputs():
        res = test_data.next_batch(batch_size)
        shared_set_d.set_value(res['dis_data'])
        shared_met_s.set_value(res['loc_data'])
        return test_model()

    # Main training routine
    return trainer.training_routine(
        model, get_train_outputs, rec_train, get_test_outputs, rec_test,
        batch_size, batch_size, train_data, test_data,
        epochs, prefix2=prefix2, check_mos_corr=False)


def run_nr_iqa(train_data, test_data, model, trainer, epochs, n_batch_imgs,
               x_c=None, mos_set=None, bat2img_idx_set=None,
               prefix2='nr_'):
    """
    @type model: .models.model_basis.ModelBasis
    @type train_data: .data_load.dataset.Dataset
    @type test_data: .data_load.dataset.Dataset
    """
    # Make dummy shared dataset
    max_num_patch = np.max(np.asarray(train_data.npat_img_list)[:, 0])
    n_pats_dummy = max_num_patch * n_batch_imgs
    sh = model.get_input_shape()
    np_set_d = np.zeros((n_pats_dummy, sh[2], sh[3], sh[1]), dtype='float32')
    shared_set_d = theano.shared(np_set_d, borrow=True)

    train_data.set_imagewise()
    test_data.set_imagewise()

    print('\nCompile theano function: Regress on MOS', end='')
    print(' (imagewise / low GPU memory)')
    if x_c is None:
        x_c = T.ftensor4('x_c')
    if mos_set is None:
        mos_set = T.vector('mos_set')
    if bat2img_idx_set is None:
        bat2img_idx_set = T.imatrix('bat2img_idx_set')

    print(' (Make training model)')
    model.set_training_mode(True)
    cost, updates, rec_train = model.cost_updates_nr_iqa(
        x_c, mos_set, n_batch_imgs, bat2img_idx_set)
    outputs = [cost] + rec_train.get_function_outputs(train=True)

    train_model = theano.function(
        [mos_set, bat2img_idx_set],
        [output for output in outputs],
        updates=updates,
        givens={
            x_c: shared_set_d
        },
        on_unused_input='warn'
    )

    print(' (Make testing model)')
    model.set_training_mode(False)
    cost, rec_test = model.cost_nr_iqa(
        x_c, mos_set, n_img=n_batch_imgs, bat2img_idx_set=bat2img_idx_set)
    outputs = [cost] + rec_test.get_function_outputs(train=False)

    test_model = theano.function(
        [mos_set, bat2img_idx_set],
        [output for output in outputs],
        givens={
            x_c: shared_set_d
        },
        on_unused_input='warn'
    )

    def get_train_outputs():
        res = train_data.next_batch(n_batch_imgs)
        np_set_d[:res['n_data']] = res['dis_data']
        shared_set_d.set_value(np_set_d)
        return train_model(res['score_set'], res['bat2img_idx_set'])

    def get_test_outputs():
        res = test_data.next_batch(n_batch_imgs)
        np_set_d[:res['n_data']] = res['dis_data']
        shared_set_d.set_value(np_set_d)
        return test_model(res['score_set'], res['bat2img_idx_set'])

    # Main training routine
    return trainer.training_routine(
        model, get_train_outputs, rec_train, get_test_outputs, rec_test,
        n_batch_imgs, n_batch_imgs, train_data, test_data,
        epochs, prefix2, check_mos_corr=True)


def make_opt_config_list(model_config):
    opt_scheme = model_config.get('opt_scheme', 'adam')
    lr = model_config.get('lr', 1e-4)

    opt_scheme_list = []
    if isinstance(opt_scheme, str):
        opt_scheme_list.append(opt_scheme)
        opt_scheme_list.append(opt_scheme)
    elif isinstance(opt_scheme, (list, tuple)):
        for c_opt_scheme in opt_scheme:
            opt_scheme_list.append(c_opt_scheme)
    else:
        raise ValueError('Improper type of opt_scheme:', opt_scheme)

    lr_list = []
    if isinstance(lr, (list, tuple)):
        for c_lr in lr:
            lr_list.append(float(c_lr))
    else:
        lr_list.append(float(lr))
        lr_list.append(float(lr))

    model_config['opt_scheme'] = opt_scheme_list[0]
    model_config['lr'] = lr_list[0]

    return opt_scheme_list, lr_list


def create_model(model_config, patch_size=None, num_ch=None):
    """
    Create a model using a model_config.
    Set input_size and num_ch according to patch_size and num_ch.
    """
    model_module_name = model_config.get('model', None)
    assert model_module_name is not None
    model_module = import_module(model_module_name)

    # set input_size and num_ch according to dataset information
    if patch_size is not None:
        model_config['input_size'] = patch_size
    if num_ch is not None:
        model_config['num_ch'] = num_ch

    return model_module.Model(model_config)
