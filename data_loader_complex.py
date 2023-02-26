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
        dict['image_complex'] = file['image_complex'][()]
        dict['data_name'] = file['image_complex'].attrs['data_name']
        dict['slice_idx'] = file['image_complex'].attrs['slice_idx']
    return dict


def load_images(data_path, h, w):
    data_path_list = sorted(glob(os.path.join(data_path, '*.h5')))

    data_array = []
    for data_idx, data_path in enumerate(data_path_list):
        img = read_h5(data_path)['image_complex']
        img = preprocess_normalisation(img)
        data_array.append(img)

    data_array = np.array(data_array).reshape((len(data_path_list), h, w))

    return data_array



if __name__ == "__main__":
    pass