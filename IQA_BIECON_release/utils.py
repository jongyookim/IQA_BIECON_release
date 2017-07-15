""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import PIL.Image as Image


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


def tile_tensor_array(X, tile_shape, img_shape=None, tile_spacing=(0, 0)):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
    tile_spacing = (int(tile_spacing[0]), int(tile_spacing[1]))
    if img_shape is None:
        img_shape = (int(X.shape[2]), int(X.shape[3]))
    else:
        img_shape = (int(img_shape[0]), int(img_shape[1]))

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                out_array[:, :, i] = (np.zeros(out_shape, dtype=dt) +
                                      channel_defaults[i])
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], tile_shape, img_shape, tile_spacing)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    this_img = this_x.reshape(img_shape)

                    # add the slice to the corresponding position in the
                    # output array
                    if Hs >= 0 and Ws >= 0:
                        out_array[
                            tile_row * (H + Hs): tile_row * (H + Hs) + H,
                            tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img

                    elif Hs < 0 and Ws < 0:
                        u_tr = int((-Hs + 1) / 2)
                        d_tr = int(-Hs / 2)
                        l_tr = int((-Ws + 1) / 2)
                        r_tr = int(-Ws / 2)
                        if tile_row == 0:
                            u_tr = 0
                        if tile_row == tile_shape[0] - 1:
                            d_tr = 0
                        if tile_col == 0:
                            l_tr = 0
                        if tile_col == tile_shape[1] - 1:
                            r_tr = 0
                        out_array[
                            tile_row * (H + Hs) + u_tr:
                            tile_row * (H + Hs) + H - d_tr,
                            tile_col * (W + Ws) + l_tr:
                            tile_col * (W + Ws) + W - r_tr
                        ] = this_img[u_tr: H - d_tr, l_tr: W - r_tr]

                    else:
                        raise NotImplementedError()
        return out_array


def tile_tensor4_from_list(X, tile_shape, idx_list=None, img_shape=None,
                           tile_spacing=(0, 0), caxis=None,
                           image_name=None):
    """
    Generate tiled image array from 4D or 3D numpy array
    Parameter
    ---------
        X : 4D or 3D numpy array
            [batch, channel, height, width] or  [batch, height, width]
    """
    assert len(X.shape) in [3, 4]
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    tile_shape = (int(tile_shape[0]), int(tile_shape[1]))
    tile_spacing = (int(tile_spacing[0]), int(tile_spacing[1]))

    if idx_list is None:
        idx_list = range(tile_shape[0] * tile_shape[1])
    else:
        assert np.max(idx_list) <= tile_shape[0] * tile_shape[1], \
            'max idx_list (%d) > number of tiles (%d)' % (
                np.max(idx_list), tile_shape[0] * tile_shape[1])

    # check image shape
    if img_shape is None:
        if len(X.shape) == 4:
            img_shape = (int(X.shape[2]), int(X.shape[3]))
            nch = int(X.shape[1])
        elif len(X.shape) == 3:
            img_shape = (int(X.shape[1]), int(X.shape[2]))
            nch = 1
        else:
            raise NotImplementedError()
    else:
        img_shape = (int(img_shape[0]), int(img_shape[1]))
        nch = int(X.shape[1])

    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if caxis is not None:
        X = image_caxis(X, caxis)
        default_rgb = [255, 0, 0]
    else:
        default_rgb = [0, 0, 0]

    # Create an output np ndarray to store the image
    out_array = np.ones((out_shape[0], out_shape[1], 3), dtype=X.dtype)
    for ch in range(3):
        out_array[:, :, ch] = out_array[:, :, ch] * default_rgb[ch]

    H, W = img_shape
    Hs, Ws = tile_spacing

    if nch == 1:
        for idx, pat_idx in enumerate(idx_list):
            this_x = X[idx]
            this_img = this_x.reshape(img_shape)

            tile_row = int(pat_idx / tile_shape[1])
            tile_col = pat_idx - tile_row * tile_shape[1]

            if Hs >= 0 and Ws >= 0:
                this_img_rgb = np.repeat(
                    this_img[:, :, np.newaxis], 3, axis=2)
                out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W,
                    :] = this_img_rgb

            elif Hs < 0 and Ws < 0:
                u_tr = int((-Hs + 1) / 2)
                d_tr = int(-Hs / 2)
                l_tr = int((-Ws + 1) / 2)
                r_tr = int(-Ws / 2)
                if tile_row == 0:
                    u_tr = 0
                if tile_row == tile_shape[0] - 1:
                    d_tr = 0
                if tile_col == 0:
                    l_tr = 0
                if tile_col == tile_shape[1] - 1:
                    r_tr = 0

                this_img_rgb = np.repeat(
                    this_img[u_tr: H - d_tr, l_tr: W - r_tr, np.newaxis],
                    3, axis=2)
                out_array[
                    tile_row * (H + Hs) + u_tr:
                    tile_row * (H + Hs) + H - d_tr,
                    tile_col * (W + Ws) + l_tr:
                    tile_col * (W + Ws) + W - r_tr,
                    :] = this_img_rgb

            else:
                raise NotImplementedError()
    elif nch == 3:
        for idx, pat_idx in enumerate(idx_list):
            for ch in range(nch):
                this_x = X[idx, ch]
                this_img = this_x.reshape(img_shape)

                tile_row = int(pat_idx / tile_shape[1])
                tile_col = pat_idx - tile_row * tile_shape[1]

                if Hs >= 0 and Ws >= 0:
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W,
                        ch] = this_img

                elif Hs < 0 and Ws < 0:
                    u_tr = int((-Hs + 1) / 2)
                    d_tr = int(-Hs / 2)
                    l_tr = int((-Ws + 1) / 2)
                    r_tr = int(-Ws / 2)
                    if tile_row == 0:
                        u_tr = 0
                    if tile_row == tile_shape[0] - 1:
                        d_tr = 0
                    if tile_col == 0:
                        l_tr = 0
                    if tile_col == tile_shape[1] - 1:
                        r_tr = 0

                    out_array[
                        tile_row * (H + Hs) + u_tr:
                        tile_row * (H + Hs) + H - d_tr,
                        tile_col * (W + Ws) + l_tr:
                        tile_col * (W + Ws) + W - r_tr,
                        ch] = this_img[u_tr: H - d_tr, l_tr: W - r_tr]

                else:
                    raise NotImplementedError()
    else:
        raise NotImplementedError()

    if image_name is not None:
        img = Image.fromarray(out_array.astype(np.uint8))
        img.save(image_name)
        return img
    else:
        return out_array


def image_from_nparray(np_arr_img, img_size=None, caxis='auto'):
    """
    Convert numpy array to PIL image
    Parameter
    ---------
        np_arr_img : 3D or 2D or 1D (img_size must be given) numpy array
            [height, width, channel] or [height, width] or [height * width]
    """
    # check img_size
    assert len(np_arr_img.shape) in [1, 2, 3]

    if len(np_arr_img.shape) == 1:
        assert img_size is not None
        if len(img_size) == 3:
            if img_size[2] == 1:
                # if gray
                img_ = np_arr_img.reshape((img_size[0], img_size[1]))
            else:
                # if RGB
                img_ = np_arr_img.reshape(img_size)
        elif len(img_size) == 2:
            if np_arr_img.shape[0] == np.product(img_size[:]):
                # if gray
                img_ = np_arr_img.reshape(img_size)
            elif np_arr_img.shape[0] == np.product(img_size[:]) * 3:
                # if RGB
                img_ = np_arr_img.reshape((img_size[0], img_size[1], 3))
            else:
                raise ValueError(
                    'Wrong shape: np_array = {0} / target = {1}'.format(
                        np_arr_img.shape, img_size))
        else:
            raise ValueError('Wrong shape: {0}'.format(img_size))
    elif len(np_arr_img.shape) == 2:
        # if gray
        img_ = np_arr_img
    else:
        if np_arr_img.shape[2] == 1:
            # if gray
            img_ = np_arr_img[:, :, 0]
        else:
            # if RGB
            assert np_arr_img.shape[2] == 3
            img_ = np_arr_img

    img_ = image_caxis(img_, caxis)
    img = Image.fromarray(img_.astype(np.uint8))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def image_from_tensor(tensor_4d, caxis='auto'):
    # transpose into (row, column, channel)
    img_ = np.transpose(tensor_4d, axes=(1, 2, 0))

    # if the image is gray, remove channel axis
    if img_.shape[2] == 1:
        img_ = img_.reshape(img_.shape[0], img_.shape[1])

    img_ = image_caxis(img_, caxis)
    img = Image.fromarray(img_.astype(np.uint8))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def image_caxis(img, caxis='auto'):
    if caxis is None or caxis == 'auto':
        min_val = img.min()
        max_val = img.max() + 1e-8
    else:
        assert len(caxis) == 2
        min_val = np.float(caxis[0])
        max_val = np.float(caxis[1])
    img = ((img - min_val) / (max_val - min_val) * 255.0).astype(img.dtype)
    img[img > 255.0] = 255.0
    img[img < 0.0] = 0.0

    return img
