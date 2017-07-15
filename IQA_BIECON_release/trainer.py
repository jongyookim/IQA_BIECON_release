from __future__ import absolute_import, division, print_function

import os
import sys
import time
import timeit
import math

import numpy as np
import PIL.Image as Image
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy.stats import kendalltau

from .utils import tile_raster_images, image_from_nparray
from .utils import tile_tensor4_from_list


class Trainer(object):
    """
    Arguments
    ---------

        model: a model to proceed training and testing. type = model instance.
        train_config: A dictionary containing:
            'batch_size': number of data in a batch
            'epochs': maximum number of epochs for training
            'test_freq': test_model the trained model every test_freq
            'save_freq': save data every save_freq
            'regular_snap_freq': save model snapshot every regular_snap_freq
            'n_imgs_to_record': number of images to record
            'prefix': prefix of filenames of recording data
        snap_path: path to save snapshot file.
        output_path: path to save output data.
    """

    def __init__(self, train_config, snap_path=None, output_path=None):

        self.test_freq = train_config.get('test_freq', None)
        assert self.test_freq is not None
        self.save_freq = train_config.get('save_freq', None)
        if self.save_freq is None:
            self.save_freq = self.test_freq
        self.regular_snap_freq = train_config.get('regular_snap_freq', 40)
        self.n_imgs_to_record = train_config.get('n_imgs_to_record', 20)

        self.prefix = train_config.get('prefix', '')
        self.set_path(snap_path, output_path)

    def set_path(self, snap_path, output_path=None):
        if snap_path is not None:
            if not os.path.isdir(snap_path):
                os.makedirs(snap_path)

        if output_path is not None:
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
        else:
            output_path = snap_path

        self.snap_path = snap_path
        self.output_path = output_path

    def training_routine(self, model, get_train_outputs, rec_train,
                         get_test_outputs, rec_test,
                         train_batch_size, test_batch_size,
                         train_data, test_data,
                         epochs, prefix2='', check_mos_corr=False):
        """
        Actual training routine.

        @type model: .models.model_basis.ModelBasis
        @type rec_train: .models.model_record.Record
        @type rec_test: .models.model_record.Record
        @type train_data: .data_load.dataset.Dataset
        @type test_data: .data_load.dataset.Dataset
        """

        # check validity
        assert self.snap_path is not None

        # get numbers of training and Testing batches
        n_train_imgs = train_data.n_data
        n_test_imgs = test_data.n_data
        n_train_batches = int(n_train_imgs / train_batch_size)
        n_test_batches = int(n_test_imgs / test_batch_size)
        assert n_train_batches > 0, 'n_train_batches = %d' % (n_train_batches)
        assert n_test_batches > 0, 'n_test_batches = %d' % (n_test_batches)

        # check n_imgs_to_record
        n_valid_rec_batches = self.n_imgs_to_record // test_batch_size + 1
        if n_valid_rec_batches > n_test_batches:
            n_valid_rec_batches = n_test_batches
        n_imgs_to_record = n_valid_rec_batches * test_batch_size

        # get numbers of data and images to monitor and write
        until_loss, until_im_info, until_img = rec_test.get_until_indices(1)

        # snapshot file names
        snapshot_file_latest = os.path.join(
            self.snap_path, self.prefix + prefix2 + 'snapshot_lastest.npy')
        snapshot_file_best = os.path.join(
            self.snap_path, self.prefix + prefix2 + 'snapshot_best.npy')
        snapshot_file_best_srcc = os.path.join(
            self.snap_path, self.prefix + prefix2 + 'snapshot_best_srcc.npy')
        snapshot_file_regular = os.path.join(
            self.snap_path, self.prefix + prefix2 + 'snapshot_{:03d}.npy')
        snapshot_file_fin = os.path.join(
            self.snap_path, self.prefix + prefix2 + 'snapshot.npy')

        # log file names
        log_file = os.path.join(
            self.snap_path, prefix2 + 'log.txt')
        log_test_file = os.path.join(
            self.output_path, prefix2 + 'log_test.txt')

        # Show information
        print('\nTrain', end='')
        if train_data.imagewise:
            print(' imagewise', end='')
        else:
            print(' patchwise', end='')
        print(' / Test', end='')
        if test_data.imagewise:
            print(' imagewise', end='')
        else:
            print(' patchwise', end='')
        print(' (%d epochs).' % (epochs))
        print('Save a snapshot every %d epochs,' % self.save_freq, end='')
        print(' and test the model every %d epochs.' % self.test_freq)
        print(' - Regular snapshot: every %d epochs' % self.regular_snap_freq)
        print(' - Snapshot path: %s' % self.snap_path)
        print(' - Batch size: %d (train) / %d (test)' % (
            train_batch_size, test_batch_size))
        print(' - Training batches: %d (%d images)' % (
            n_train_batches, n_train_imgs))
        print(' - Testing batches: %d (%d images)' % (
            n_test_batches, n_test_imgs), end='')
        print(' / Missed images: %d' % (
            n_test_imgs - n_test_batches * test_batch_size))
        print(' - Monitor data: %s' % (', '.join(rec_train.data_keys)))
        print(' - Monitor images: %s' % (', '.join(rec_test.data_keys)))
        print(' - Monitor im. data: %s' % (', '.join(rec_test.im_data_keys)))
        print(' - Num of rec. images: %d (%d x %d batches)' % (
            n_imgs_to_record, test_batch_size, n_valid_rec_batches))

        # get MOS list
        if check_mos_corr:
            # if check_mos_corr is true, the first value of
            # rec_im_data must be mos predicted.
            assert rec_test.im_data_keys[0] == 'mos_p'
            assert test_data.exist_score
            n_valid_test_imgs = n_test_batches * test_batch_size
            test_score_list = test_data.score_data[:n_valid_test_imgs]
            mos_p_list = np.zeros(n_valid_test_imgs, dtype='float32')
            print(' - Check SRCC/PLCC using %d images' % (n_valid_test_imgs))

        start_time = timeit.default_timer()
        prev_time = start_time
        best_test_loss = np.inf

        # write current time in log file
        cur_time = 'Started at %s\n' % (time.strftime('%X %x'))
        key_str = 'cost, ' + ", ".join(rec_train.data_keys) + '\n'
        with open(log_file, 'a') as f_hist:
            f_hist.write(cur_time)
            f_hist.write(key_str)

        key_str = 'cost, ' + ", ".join(rec_train.data_keys)
        key_str += ', SRCC, PLCC\n' if check_mos_corr else '\n'
        with open(log_test_file, 'a') as f_hist:
            f_hist.write(cur_time)
            f_hist.write(key_str)

        best_score_set = (0., 0., -1) if check_mos_corr else (np.inf, 0., -1)

        # go through training epochs
        for epoch in range(epochs):
            # train model
            losses = np.zeros(rec_train.num_data + 1, dtype='float32')
            for batch_idx in range(n_train_batches):
                # get training loss
                losses += get_train_outputs()
            losses /= n_train_batches

            # write log
            with open(log_file, 'a') as f_hist:
                data = '%d' % (epoch + 1)
                for idx in range(-1, rec_train.num_data):
                    data += '\t%.6f' % (losses[idx + 1])
                data += '\n'
                f_hist.write(data)

            # show information
            end_time = timeit.default_timer()
            pr_str = ' {:3d}, cost {:.3f}, '.format(epoch + 1, losses[0])
            for idx, key in enumerate(rec_train.data_keys):
                pr_str += '{:s} {:.3f}, '.format(key, losses[idx + 1])
            minutes, seconds = divmod(end_time - prev_time, 60)
            pr_str += 'time {:02.0f}:{:05.2f}\n'.format(minutes, seconds)
            sys.stdout.write(pr_str)
            sys.stdout.flush()
            prev_time = end_time

            if (epoch + 1) % self.regular_snap_freq == 0:
                model.save(snapshot_file_regular.format(epoch + 1))

            ##################################################################
            # test_model the trained model and save a snapshot
            # For every safe_freq and test_freq
            test_model = (epoch + 1) % self.test_freq == 0
            save_data = (epoch + 1) % self.save_freq == 0
            if test_model or save_data:
                if save_data:
                    # make output folder
                    numstr = '{:03d}'.format(epoch + 1)
                    out_path = os.path.join(
                        self.output_path, prefix2 + numstr + '/')
                    if not os.path.isdir(out_path):
                        os.makedirs(out_path)

                    im_data = np.zeros(
                        (rec_test.num_im_data, n_imgs_to_record),
                        dtype='float32')

                losses = np.zeros(rec_test.num_data + 1, dtype='float32')
                for test_bat_idx in range(0, n_test_batches):
                    # get testing loss
                    outputs = get_test_outputs()
                    losses += outputs[:until_loss]
                    cur_im_data = outputs[until_loss:until_im_info]
                    cur_images = outputs[until_im_info:until_img]

                    # get predicted mos
                    if check_mos_corr:
                        mos_p = cur_im_data[0]
                        idx_from = test_bat_idx * test_batch_size
                        idx_to = (test_bat_idx + 1) * test_batch_size
                        mos_p_list[idx_from:idx_to] = mos_p

                    # write image data
                    if (save_data and rec_test.num_im_data > 0 and
                            test_bat_idx < n_valid_rec_batches):
                        idx_from = test_bat_idx * test_batch_size
                        idx_to = (test_bat_idx + 1) * test_batch_size
                        im_data[:, idx_from:idx_to] = cur_im_data

                    # write images
                    if (save_data and rec_test.num_imgs > 0 and
                            test_bat_idx < n_valid_rec_batches):
                        if test_data.imagewise:
                            rec_info = test_data.get_current_recon_info()
                            draw_tiled_images(
                                cur_images, rec_test.rec_imgs, test_bat_idx,
                                out_path,
                                rec_info['bat2img_idx_set'],
                                rec_info['npat_img_list'],
                                rec_info['filt_idx_list'],
                                test_data.patch_size,
                                test_data.patch_step)
                        else:
                            draw_images(
                                cur_images, rec_test.rec_imgs, test_bat_idx,
                                test_batch_size, out_path)

                losses /= n_test_batches

                # get SRCC and PLCC
                if check_mos_corr:
                    rho_s, _ = spearmanr(test_score_list, mos_p_list)
                    rho_p, _ = pearsonr(test_score_list, mos_p_list)

                    if math.isnan(rho_s) or math.isnan(rho_p):
                        print('@ Stop iteration! (NaN)')
                        best_score_set = (0, 0, epoch)
                        break
                    else:
                        if rho_s > best_score_set[0]:
                            best_score_set = (rho_s, rho_p, epoch)
                            model.save(snapshot_file_best_srcc)
                else:
                    if losses[0] < best_score_set[0]:
                        if rec_test.num_data >= 1:
                            best_score_set = (losses[0], losses[1], epoch)
                        else:
                            best_score_set = (losses[0], 0, epoch)

                # save the latest snapshot
                model.save(snapshot_file_latest)

                # save the best snapshot
                if losses[0] < best_test_loss:
                    best_test_loss = losses[0]
                    print(' # BEST', end=' ')
                    model.save(snapshot_file_best)

                # For every save_freq
                if save_data:
                    # write image data
                    if rec_test.num_im_data > 0:
                        with open(out_path + 'info.txt', 'w') as f:
                            # header
                            data = 'epoch: %s (%s)\n' % (
                                numstr, ', '.join(rec_test.im_data_keys))
                            f.write(data)

                            for idx in range(n_imgs_to_record):
                                imidx = idx
                                data = '%d' % idx
                                for ii in range(rec_test.num_im_data):
                                    data += '\t%.6f' % (im_data[ii][imidx])
                                data += '\n'
                                f.write(data)

                    # write mos
                    if check_mos_corr:
                        with open(out_path + 'mos_res.txt', 'w') as f:
                            # header
                            data = 'epoch: %s (mos_p, mos)\n' % (numstr)
                            f.write(data)

                            for idx in range(n_valid_test_imgs):
                                data = '{:.6f}\t{:.6f}\n'.format(
                                    mos_p_list[idx], test_score_list[idx])
                                f.write(data)
                            data = 'SRCC: {:.4f}, PLCC: {:.4f}\n'.format(
                                rho_s, rho_p)
                            f.write(data)

                    # write kernel images
                    draw_kernels(rec_test.rec_kernels, self.output_path,
                                 prefix2, '_' + numstr)

                # write log
                with open(log_test_file, 'a') as f_hist:
                    data = '{:d}'.format(epoch + 1)
                    for idx in range(-1, rec_test.num_data):
                        data += '\t{:.6f}'.format(losses[idx + 1])
                    if check_mos_corr:
                        data += '\t{:.4f}\t{:.4f}'.format(rho_s, rho_p)
                    data += '\n'
                    f_hist.write(data)

                # show information
                end_time = timeit.default_timer()
                pr_str = ' * vcost {:.3f}, '.format(losses[0])
                for idx, key in enumerate(rec_train.data_keys):
                    pr_str += '{:s} {:.3f}, '.format(key, losses[idx + 1])
                if check_mos_corr:
                    pr_str += 'SRCC {:.3f}, PLCC {:.3f}, '.format(rho_s, rho_p)
                minutes, seconds = divmod(end_time - prev_time, 60)
                pr_str += 'time {:02.0f}:{:05.2f}\n'.format(minutes, seconds)
                sys.stdout.write(pr_str)
                sys.stdout.flush()
                prev_time = end_time

        end_time = timeit.default_timer()
        total_time = end_time - start_time
        print(' - Train ran for %.2fm' % ((total_time) / 60.))
        print(' - Finished at %s' % (time.strftime('%X %x')))

        if best_score_set[0] != 0:
            model.save(snapshot_file_fin)

        return best_score_set

    def testing_routine(self, get_test_outputs, rec_test,
                        test_batch_size, test_data, prefix2='',
                        check_mos_corr=False):
        """Actual testing routine: group patches for each image

        @type rec_test: .models.model_record.Record
        """
        # get numbers of training and Testing batches
        n_test_imgs = test_data.n_images
        n_test_batches = int(n_test_imgs / test_batch_size)
        assert n_test_batches > 0

        n_valid_test_imgs = n_test_batches * test_batch_size

        if self.n_imgs_to_record == 'all':
            n_imgs_to_record = n_valid_test_imgs
        else:
            n_valid_rec_batches = self.n_imgs_to_record // test_batch_size + 1
            if n_valid_rec_batches > n_test_batches:
                n_valid_rec_batches = n_test_batches
            n_imgs_to_record = n_valid_rec_batches * test_batch_size

        # get numbers of data and images to monitor and write
        until_loss = rec_test.num_data + 1
        until_im_info = until_loss + rec_test.num_im_data
        until_img = until_im_info + rec_test.num_imgs

        # Show information
        print('\nTest the model')
        if test_data.imagewise:
            print(' (imagewise)')
        else:
            print(' (patchwise)')
        print(' - Num of images in a batch: %d' % (test_batch_size))
        print(' - Testing batches: %d (%d images)' % (
            n_test_batches, n_test_imgs))
        print(' - Missed images in validation: %d' % (
            n_test_imgs - n_test_batches * test_batch_size))
        print(' - Image recording batches: %d (%d images)' % (
            n_valid_rec_batches, n_imgs_to_record))
        print(' - Monitor data: %s' % (', '.join(rec_test.data_keys)))
        print(' - Monitor images: %s' % (', '.join(rec_test.data_keys)))
        print(' - Monitor im. data: %s' % (', '.join(rec_test.im_data_keys)))

        # get MOS list
        if check_mos_corr:
            # if check_mos_corr is true, the first value of
            # rec_im_data must be mos predicted.
            assert rec_test.im_data_keys[0] == 'mos_p'
            assert test_data.exist_score

            test_score_list = test_data.score_data[:n_valid_test_imgs]
            mos_p_list = np.zeros(n_valid_test_imgs, dtype='float32')
            print(' - Check SRCC/PLCC using %d images' % (n_valid_test_imgs))

        start_time = timeit.default_timer()
        prev_time = start_time

        # write current time in log file
        cur_time = 'Started at %s\n' % (time.strftime('%X %x'))
        log_file = os.path.join(self.output_path, prefix2 + 'log_test.txt')
        with open(log_file, 'a') as f_hist:
            f_hist.write(cur_time)

        out_path = os.path.join(self.output_path, prefix2 + '/')
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        im_data = np.zeros(
            (rec_test.num_im_data, n_valid_test_imgs), dtype='float32')

        best_score_set = (0., 0.) if check_mos_corr else (np.inf, np.inf)

        losses = np.zeros(rec_test.num_data + 1, dtype='float32')
        for test_bat_idx in range(0, n_test_batches):
            # get testing loss
            outputs = get_test_outputs()

            losses += outputs[:until_loss]
            cur_im_data = outputs[until_loss:until_im_info]
            cur_images = outputs[until_im_info:until_img]

            # get predicted mos
            if check_mos_corr:
                mos_p = cur_im_data[0]
                idx_from = test_bat_idx * test_batch_size
                idx_to = (test_bat_idx + 1) * test_batch_size
                mos_p_list[idx_from:idx_to] = mos_p

            # write image data
            if rec_test.num_im_data > 0:
                idx_from = test_bat_idx * test_batch_size
                idx_to = (test_bat_idx + 1) * test_batch_size
                im_data[:, idx_from:idx_to] = cur_im_data

            # write images
            if rec_test.num_imgs > 0 and test_bat_idx < n_valid_rec_batches:
                if test_data.imagewise:
                    rec_info = test_data.get_current_recon_info()
                    draw_tiled_images(
                        cur_images, rec_test.rec_imgs, test_bat_idx,
                        out_path,
                        rec_info['bat2img_idx_set'],
                        rec_info['npat_img_list'],
                        rec_info['filt_idx_list'],
                        test_data.patch_size,
                        test_data.patch_step)
                else:
                    draw_images(
                        cur_images, rec_test.rec_imgs, test_bat_idx,
                        test_batch_size, out_path)
                rec_info = test_data.get_current_recon_info()
                draw_tiled_images(
                    cur_images, rec_test.rec_imgs, test_bat_idx, out_path,
                    rec_info['bat2img_idx_set'],
                    rec_info['npat_img_list'],
                    rec_info['filt_idx_list'],
                    test_data.patch_size,
                    test_data.patch_step)

        losses /= n_test_batches

        # get SRCC and PLCC
        if check_mos_corr:
            rho_s, _ = spearmanr(test_score_list, mos_p_list)
            rho_p, _ = pearsonr(test_score_list, mos_p_list)
            tau, _ = kendalltau(test_score_list, mos_p_list)
            rmse = np.sqrt(((test_score_list - mos_p_list) ** 2).mean())
            best_score_set = (rho_s, rho_p)
        else:
            if rec_test.num_data >= 1:
                best_score_set = (losses[0], losses[1])
            else:
                best_score_set = (losses[0], 0)

        # write image data
        if rec_test.num_im_data > 0:
            with open(out_path + 'info.txt', 'w') as f:
                # header
                data = 'imidx, %s\n' % (
                    ', '.join(rec_test.im_data_keys))
                f.write(data)

                for idx in range(n_valid_test_imgs):
                    imidx = idx
                    data = '%d' % idx
                    for ii in range(rec_test.num_im_data):
                        data += '\t%.6f' % (im_data[ii][imidx])
                    data += '\n'
                    f.write(data)

        # write mos
        if check_mos_corr:
            with open(out_path + 'mos_res.txt', 'w') as f:
                # header
                data = 'mos_p, mos\n'
                f.write(data)

                for idx in range(n_valid_test_imgs):
                    data = '{:.6f}\t{:.6f}\n'.format(
                        mos_p_list[idx], test_score_list[idx])
                    f.write(data)
                data = 'SRCC: {:.4f}, PLCC: {:.4f}'.format(rho_s, rho_p)
                data += ', KRCC: {:.4f}, RMSE: {:.4f}\n'.format(tau, rmse)
                f.write(data)

        # write kernel images
        draw_kernels(rec_test.rec_kernels, self.output_path, prefix2)

        # show information
        end_time = timeit.default_timer()
        pr_str = ' * vcost {:.3f}, '.format(losses[0])
        for idx, key in enumerate(rec_test.data_keys):
            pr_str += '{:s} {:.3f}, '.format(key, losses[idx + 1])
        if check_mos_corr:
            pr_str += 'SRCC {:.3f}, PLCC {:.3f}, '.format(rho_s, rho_p)
            pr_str += 'KRCC {:.3f}, RMSE {:.3f}, '.format(tau, rmse)
        minutes, seconds = divmod(end_time - prev_time, 60)
        pr_str += 'time {:02.0f}:{:05.2f}\n'.format(minutes, seconds)
        sys.stdout.write(pr_str)
        sys.stdout.flush()
        prev_time = end_time

        end_time = timeit.default_timer()
        total_time = end_time - start_time
        print(' - Test ran for %.2fm' % ((total_time) / 60.))
        print(' - Finished at %s' % (time.strftime('%X %x')))

        return best_score_set


def draw_kernels(kernels, out_path, prefix='', suffix=''):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for idx in range(len(kernels)):
        kernel = kernels[idx].get_value(borrow=True)
        name = kernels[idx].name.replace('/', '_')
        assert len(kernel.shape) == 4
        (nkern, nfeat, kern_sz0, kern_sz1) = kernel.shape
        tile = int(np.ceil(np.sqrt(nkern)))

        imgshape = ((kern_sz0 + 1) * tile - 1, (kern_sz1 + 1) * tile - 1)
        tot_kern_array = np.zeros((nfeat, imgshape[0] * imgshape[1]))
        feat_tile = int(np.ceil(np.sqrt(nfeat)))

        for fidx in range(nfeat):
            kern_array = tile_raster_images(
                X=kernel[:, fidx, :, :],
                img_shape=(kern_sz0, kern_sz1),
                tile_shape=(tile, tile),
                tile_spacing=(1, 1))
            tot_kern_array[fidx] = kern_array.flatten()

        tot_kern_image = Image.fromarray(tile_raster_images(
            X=tot_kern_array,
            img_shape=imgshape,
            tile_shape=(feat_tile, feat_tile),
            tile_spacing=(2, 2)))

        img_name = '%s%s%s.png' % (prefix, name, suffix)
        tot_kern_image.save(os.path.join(out_path, img_name))


def draw_tiled_images(images, img_info_dict, bat_idx, out_path,
                      bat2img_idx_set, npat_img_list, filt_idx_list=None,
                      patch_size=None, patch_step=None):
    n_batch_imgs = len(npat_img_list)

    for ii, key in enumerate(img_info_dict):
        for idx in range(n_batch_imgs):
            idx_from, idx_to = bat2img_idx_set[idx]
            cur_img = images[ii][idx_from: idx_to]
            caxis = img_info_dict[key].get('caxis', None)
            scale = img_info_dict[key].get('scale', None)
            if scale:
                tile_spacing = (
                    int(-(patch_size[0] - patch_step[0]) * scale),
                    int(-(patch_size[1] - patch_step[1]) * scale))
            else:
                tile_spacing = (0, 0)

            nch = int(cur_img.shape[1])
            if nch == 1 or nch == 3:
                tiled_array = tile_tensor4_from_list(
                    X=cur_img,
                    tile_shape=npat_img_list[idx][1:],
                    idx_list=filt_idx_list[idx],
                    tile_spacing=tile_spacing,
                    caxis=caxis)
                img = Image.fromarray(tiled_array.astype(np.uint8))
                img_name = '%d_%s.png' % (bat_idx * n_batch_imgs + idx, key)
                img.save(os.path.join(out_path, img_name))
            else:
                for ch_idx in range(nch):
                    tiled_array = tile_tensor4_from_list(
                        X=cur_img[:, ch_idx, :, :],
                        tile_shape=npat_img_list[idx][1:],
                        idx_list=filt_idx_list[idx],
                        tile_spacing=tile_spacing,
                        caxis=caxis)
                    img = Image.fromarray(tiled_array.astype(np.uint8))
                    img_name = '%d_%s_%02d.png' % (
                        bat_idx * n_batch_imgs + idx, key, ch_idx)
                    img.save(os.path.join(out_path, img_name))


def draw_images(images, img_info_dict, bat_idx, n_batch_imgs, out_path):
    for ii, key in enumerate(img_info_dict):
        for idx in range(n_batch_imgs):
            cur_img = images[ii][idx]
            caxis = img_info_dict[key].get('caxis', None)

            nch = int(cur_img.shape[0])
            if nch == 1 or nch == 3:
                img = image_from_nparray(
                    np.transpose(cur_img, (1, 2, 0)), caxis=caxis)
                img_name = '%d_%s.png' % (bat_idx * n_batch_imgs + idx, key)
                img.save(os.path.join(out_path, img_name))
            else:
                for ch_idx in range(nch):
                    img = image_from_nparray(
                        cur_img[ch_idx, :, :], caxis=caxis)
                    img_name = '%d_%s_%02d.png' % (
                        bat_idx * n_batch_imgs + idx, key, ch_idx)
                    img.save(os.path.join(out_path, img_name))
