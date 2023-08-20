import os
from pickle import load, dump
import cv2
import gc
import numpy as np
import scipy.io as scio
import h5py
from glob import glob


def preprocess_normalisation(img):
    img = img / abs(img).max()

    return img


def read_h5(data_path):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        dict['espirit_complex'] = file['espirit_complex'][()]
        dict['filename'] = file['espirit_complex'].attrs['filename']
        dict['data_type'] = file['espirit_complex'].attrs['data_type']
        dict['case_idx'] = file['espirit_complex'].attrs['case_idx']
        dict['slice_idx'] = file['espirit_complex'].attrs['slice_idx']
        # dict['rss'] = file['rss'][()]

    return dict


def load_images(data_path, h, w, debug=False):
    data_path_list = sorted(glob(os.path.join(data_path, '*.h5')))
    if debug:
        data_path_list = data_path_list[0:100]
    data_array = []
    data_info = []
    for data_idx, data_path in enumerate(data_path_list):
        data_dict = read_h5(data_path)
        img = data_dict['espirit_complex']
        img = preprocess_normalisation(img)
        # filename = data_dict['filename']
        data_type = data_dict['data_type']
        case_idx = data_dict['case_idx']
        slice_idx = data_dict['slice_idx']
        slice_info = '{}_{:02d}_{:03d}'.format(data_type, case_idx, slice_idx)

        data_array.append(img)
        data_info.append(slice_info)

    data_array = np.array(data_array).reshape((len(data_path_list), h, w))

    return data_array, data_info



if __name__ == "__main__":
    pass