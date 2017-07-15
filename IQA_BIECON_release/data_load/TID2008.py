from __future__ import absolute_import, division, print_function
import os
import numpy as np

# Define DB information
BASE_PATH = 'D:/DB/IQA/TID2008'
LIST_FILE_NAME = 'TID2008.txt'
ALL_SCENES = list(range(24))
# ALL_SCENES = list(range(25))
ALL_DIST_TYPES = list(range(17))


def make_image_list(scenes, dist_types=None, show_info=True):
    """
    Make image list from TID2008 database
    TID2008: 25 reference images x 17 distortions x 4 levels
    """
    # Get reference / distorted image file lists:
    # d_img_list and score_list
    d_img_list, r_img_list, score_list = [], [], []
    list_file_name = os.path.join(BASE_PATH, LIST_FILE_NAME)
    with open(list_file_name, 'r') as listFile:
        for line in listFile:
            # ref_idx ref_name dist_name dist_types, DMOS
            (scn_idx, dis_idx, ref, dis, score) = line.split()
            scn_idx = int(scn_idx)
            dis_idx = int(dis_idx)
            if scn_idx in scenes and dis_idx in dist_types:
                d_img_list.append(dis)
                r_img_list.append(ref)
                score_list.append(float(score))

    score_list = np.array(score_list, dtype='float32')
    n_images = len(d_img_list)

    if show_info:
        print(' - Scenes: %s' % ', '.join([str(i) for i in scenes]))
        print(' - Distortion types: %s' % ', '.join(
            [str(i) for i in dist_types]))
        print(' - Number of images: {:,}'.format(n_images))
        print(' - MOS range: [{:.2f}, {:.2f}]'.format(
            np.min(score_list), np.max(score_list)))

    return {
        'scenes': scenes,
        'dist_types': dist_types,
        'base_path': BASE_PATH,
        'n_images': n_images,
        'd_img_list': d_img_list,
        'r_img_list': r_img_list,
        'score_list': score_list}

