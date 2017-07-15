from __future__ import absolute_import, division, print_function

from collections import OrderedDict


class Record(object):

    def __init__(self):
        self.rec_data = OrderedDict()
        self.rec_im_data = OrderedDict()
        self.rec_imgs = OrderedDict()
        self.rec_kernels = []

    ###########################################################################
    # Functions for recording data

    @property
    def data_keys(self):
        """Get dictionary keys of `rec_data`"""
        return list(self.rec_data)

    @property
    def im_data_keys(self):
        """Get dictionary keys of `rec_im_data`"""
        return list(self.rec_im_data)

    @property
    def imgs_keys(self):
        """Get dictionary keys of `rec_imgs`"""
        return list(self.rec_imgs)

    @property
    def num_data(self):
        """Get number of `rec_data`"""
        return len(self.rec_data)

    @property
    def num_im_data(self):
        """Get number of `rec_im_data`"""
        return len(self.rec_im_data)

    @property
    def num_imgs(self):
        """Get number of `rec_imgs`"""
        return len(self.rec_imgs)

    def empty_records(self):
        self.rec_data.clear()
        self.rec_im_data.clear()
        self.rec_imgs.clear()
        self.rec_kernels = []

    def add_data(self, name, data, **kwargs):
        """Add scalar data of one minibatcth to monitor.
        """
        kwargs['data'] = data
        self.rec_data[name] = kwargs

    def add_im_data(self, name, data, **kwargs):
        """Add scalar data for each image (imagewise) or patch (patchwise)
        to record.
        """
        kwargs['data'] = data
        self.rec_im_data[name] = kwargs

    def add_imgs(self, name, data, **kwargs):
        """Add image data for each image (imagewise) or patch (patchwise)
        to record.
        Supplementary information can be added via `**kwargs`.
        """
        kwargs['data'] = data
        self.rec_imgs[name] = kwargs

    def get_function_outputs(self, train=False):
        if train:
            return (self.get_data())
        else:
            return (self.get_data() + self.get_im_data() + self.get_imgs())

    def get_data(self):
        return [elem['data'] for elem in list(self.rec_data.values())]

    def get_im_data(self):
        return [elem['data'] for elem in list(self.rec_im_data.values())]

    def get_imgs(self):
        return [elem['data'] for elem in list(self.rec_imgs.values())]

    def get_until_indices(self, start=1):
        """Returns the 'until-indices' for each reording data type.
        """
        until_loss = len(self.rec_data) + start
        until_im_info = until_loss + len(self.rec_im_data)
        until_img = until_im_info + len(self.rec_imgs)
        return until_loss, until_im_info, until_img

    def add_kernel(self, layers, nth_layers):
        """Add a kernel image from the `nth_layers` of self.layers[`key`]
        to record.
        """
        if isinstance(nth_layers, (list, tuple)):
            for nth in nth_layers:
                layer = layers[nth]
                assert layer.__class__.__name__ == 'ConvLayer'
                self.rec_kernels.append(layer.W)
        else:
            layer = layers[nth_layers]
            assert layer.__class__.__name__ == 'ConvLayer'
            self.rec_kernels.append(layer.W)

    # def get_rec_info(self):
    #     rec_info = {}
    #     rec_info['rec_data'] = self.exclude_info(self.rec_data, 'data')
    #     rec_info['rec_im_data'] = self.exclude_info(self.rec_im_data, 'data')
    #     rec_info['rec_imgs'] = self.exclude_info(self.rec_imgs, 'data')
    #     return rec_info

    # def exclude_info(self, dic, exclude):
    #     new_dic = OrderedDict()
    #     for dic_key in dic:
    #         new_elems = {}
    #         for elem_key in dic[dic_key]:
    #             if elem_key == exclude:
    #                 continue
    #             else:
    #                 new_elems[elem_key] = dic[dic_key][elem_key]
    #         new_dic[dic_key] = new_elems
    #     return new_dic
