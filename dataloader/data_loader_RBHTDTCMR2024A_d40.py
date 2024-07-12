'''
# -----------------------------------------
Data Loader
DatasetRBHTDTCMR2024A d.4.0
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''
import torch.utils.data as data
# import utils.utils_image as util
# from utils.utils_fourier import *
# from models.select_mask import define_Mask
# from utils.utils_kspace_undersampling.utils_undersampling_pattern import *
import os
import csv
import h5py
import numpy as np
import pandas as pd
# import pickle5 as pickle

def preprocess_normalisation(img, max_value=None, min_value=None):

    if max_value is None:
        max_value = img.max()
    if min_value is None:
        min_value = img.min()

    img = (img - min_value) / (max_value - min_value)

    return img


def preprocess_remove_outlier(img, threshold_down=1, threshold_up=99):

    img = np.clip(img, np.percentile(img, threshold_down), np.percentile(img, threshold_up))

    return img

def preprocess_shape(img):

    # (h, w) --> (256, ?)
    assert len(img.shape) == 2  # check 2D
    init_h, init_w, = img.shape
    if init_h > init_w:
        pass
    elif init_h < init_w:
        img = img.transpose(1, 0)
    else:
        raise ValueError
    trans_h, trans_w, = img.shape
    assert trans_h == 256
    assert trans_h > trans_w

    # (256, ?) --> (256, 96)
    if trans_w < 96:
        # padding
        pad_l = (96 - trans_w) // 2
        pad_r = 96 - trans_w - pad_l
        img = np.pad(img, ((0, 0), (pad_l, pad_r)), 'constant', constant_values=(0, 0))
    elif trans_w > 96:
        raise ValueError
        # cropping, but should not happen in this dataset
        # crop_l = (trans_w - 96) // 2
        # crop_r = trans_w - 96 - crop_l
        # img = img[:, crop_l: crop_l+96]
    else:
        pass

    return img[:, :, np.newaxis]

def read_dwi(case_path, slice_name):

    data = pd.read_hdf(os.path.join(case_path, 'data.h5'), 'data')
    # print(data.keys())

    data_dict = {}
    for key in data.keys():
        data_dict[key] = [data[key][int(slice_name)]]
    return data_dict


def read_dt(case_path):

    data = np.load(os.path.join(case_path, 'results', 'results.npz'))

    data_dict = {}
    for key in data.files:
        data_dict[key] = data[key]

    return data_dict


def read_mask(case_path):

    data = np.load(os.path.join(case_path, 'mask_3c.npz'), allow_pickle=True)

    data_dict = {}
    for key in data.files:
        data_dict[key] = data[key]

    return data_dict


def load_slice(case_path, slice_name):
    dwi_dict = read_dwi(case_path, slice_name)
    dt_dict = read_dt(case_path)
    mask_dict = read_mask(case_path)
    data_dict = {'dwi': dwi_dict, 'dt': dt_dict, 'mask': mask_dict}
    return data_dict


def generate_gaussian_noise(x, noise_level, noise_var):
    spower = np.sum(x ** 2) / x.size
    npower = noise_level / (1 - noise_level) * spower
    noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
    return noise


def undersample_kspace(x, mask, is_noise, noise_level, noise_var):

    fft = fft2(x[:, :, 0])
    fft = fftshift(fft)
    fft = fft * mask
    if is_noise:
        fft = fft + generate_gaussian_noise(fft, noise_level, noise_var)
    fft = ifftshift(fft)
    xx = ifft2(fft)
    xx = np.abs(xx)

    x = xx[:, :, np.newaxis]

    return x


def load_images(dataroot, log_folder_path, h, w, phase='train', disease='', cphase='', debug=False):

    data_infos = []
    case_infos = []

    dataroot = dataroot
    log_folder_path = log_folder_path

    disease = disease
    cphase = cphase

    k_fold_cross_validation = 5
    fold_index_train = [0, 1, 2, 3, 4]
    fold_index_val = []
    fold_index_pool = list(np.linspace(0, k_fold_cross_validation, k_fold_cross_validation, endpoint=False, dtype=np.int))
    assert set(fold_index_train + fold_index_val) == set(fold_index_pool)

    n_dwi_preset = None
    get_dt_original = None
    dwi_shuffle = None
    in_memory = True
    remove_bad_frames = False

    case_list = []
    slice_list = []
    slice_valid_list = []

    # load fold index list for train val cross validation
    if phase == 'train':
        fold_index = fold_index_train
    elif phase == 'val':
        fold_index = fold_index_val
    elif phase == 'test':
        fold_index = [0]

    # record case
    for idx in fold_index:
        for cp in cphase:
            if phase == 'train' or phase == 'val':
                record_path = 'TrainVal-F{}_{}_{}_Slice.csv'.format(idx, disease, cp)
            elif phase == 'test':
                record_path = 'Test_{}_{}_Slice.csv'.format(disease, cp)
            with open(os.path.join(dataroot, log_folder_path, record_path), 'r', encoding="UTF-8") as csvfile:
                reader = csv.reader(csvfile)
                cases = []
                slices = []
                slice_valid = []
                for row in reader:
                    cases.append(row[0])
                    slices.append(row[1])
                    slice_valid.append(bool(int(row[2])))
            case_list += cases
            slice_list += slices
            slice_valid_list += slice_valid
    assert slice_list, 'No data loaded.'

    if remove_bad_frames:
        slice_list = [slice_list[i] for i in range(len(slice_valid_list)) if slice_valid_list[i] is not True]
        case_list = [case_list[i] for i in range(len(slice_valid_list)) if slice_valid_list[i] is not True]
        slice_valid_list_check = [slice_valid_list[i] for i in range(len(slice_valid_list)) if slice_valid_list[i] is not True]
        assert len(slice_list) == len(case_list) == len(slice_valid_list_check)
        assert all([i is False for i in slice_valid_list_check])


    data_array = []
    data_info = []

    for idx_slice, slice_name in enumerate(slice_list):
        if (idx_slice > 30) and debug:
            break

        case_name = case_list[idx_slice]
        slice_name = slice_list[idx_slice]
        slice_valid = slice_valid_list[idx_slice]
        case_path = os.path.join(dataroot, case_name)
        data_dict = load_slice(case_path, slice_name)

        img_info = '{}_{}'.format(case_name, slice_name)
        info = [case_name, slice_name]

        dwi_image = np.stack(data_dict['dwi']['image'], axis=0)  # (n_dwi, h, w)
        n_dwi = dwi_image.shape[0]
        assert n_dwi == 1

        img_H = dwi_image[0, ...]

        # preprocessing: normalisation ?~? --> 0~1
        img_H = preprocess_normalisation(img_H)

        # preprocessing: shape (?, ?) --> (256, 96)
        img_H = preprocess_shape(img_H)

        # (256, 96, 1) --> (256, 96)
        img_H = img_H[..., 0]

        data_array.append(img_H)
        data_info.append(info)

    data_array = np.array(data_array)

    return data_array, data_info



if __name__ == "__main__":
    pass