import os
from pickle import load, dump
import cv2
import gc
import numpy as np
import scipy.io as scio
import h5py
from glob import glob
import csv


def preprocess_normalisation(img, max_value=None, min_value=None):

    if max_value is None:
        max_value = img.max()
    if min_value is None:
        min_value = img.min()

    img = (img - min_value) / (max_value - min_value)

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

def read_h5(data_path, case_name, slice_name):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        # data for specific case and slice
        data = file['images'][slice_name]

        dict['pixel_array'] = data[()].astype(complex)

        dict['data_info'] = {}
        dict['data_info']['dcmFileName'] = data.attrs['dcmFileName']  # DICOM file name
        dict['data_info']['dcmFilePath'] = data.attrs['dcmFilePath']  # DICOM file path
        dict['data_info']['caseName'] = case_name
        dict['data_info']['slice_name'] = slice_name

        #TODO: add more info we need here.
        #TODO: the physics information now is str format, but we need to change it to other meaningful format (float, array).
        dict['phys_info'] = {}
        dict['phys_info']['bvalue'] = data.attrs['(0019, 100C)'] if '(0019, 100C)' in data.attrs.keys() else None  # b-value
        dict['phys_info']['Gradient Direction'] = data.attrs['(0019, 100E)'] if '(0019, 100E)' in data.attrs.keys() else None  # gradient direction
        dict['phys_info']['bspoil'] = data.attrs['(0019, 4000)'] if '(0019, 4000)' in data.attrs.keys() else None  # bspoil
        dict['phys_info']['ImageOrientationPatient'] = data.attrs['(0020, 0037)'] if '(0020, 0037)' in data.attrs.keys() else None  # Image Orientation Patient
        dict['phys_info']['Nominal Interval'] = data.attrs['(0018, 1062)'] if '(0018, 1062)' in data.attrs.keys() else None  # Nominal Interval

    return dict


def load_images(dataroot, h, w, phase='train', disease='', cphase='', debug=False):

    data_infos = []
    case_infos = []

    if disease == 'MI':
        log_name = 'log_mi'
    else:
        log_name = 'log'

    if phase == 'train':
        fold_index = [1, 2, 3, 4]

        # record slice
        for idx in fold_index:
            record_path = 'TrainVal-F{}_{}_{}_Slice.csv'.format(idx, disease, cphase)
            with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
            data_infos += rows
        assert data_infos, 'No data loaded.'

        # record case
        for idx in fold_index:
            record_path = 'TrainVal-F{}_{}_{}_Case.csv'.format(idx, disease, cphase)
            with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row[0] for row in reader]
            case_infos += rows
        assert case_infos, 'No data loaded.'

    elif phase == 'val':
        fold_index = [0]

        # record slice
        for idx in fold_index:
            record_path = 'TrainVal-F{}_{}_{}_Slice.csv'.format(idx, disease, cphase)
            with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
            data_infos += rows
        assert data_infos, 'No data loaded.'

        # record case
        for idx in fold_index:
            record_path = 'TrainVal-F{}_{}_{}_Case.csv'.format(idx, disease, cphase)
            with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
                reader = csv.reader(csvfile)
                rows = [row[0] for row in reader]
            case_infos += rows
        assert case_infos, 'No data loaded.'

    elif phase == 'test':
        # record slice
        record_path = 'Test_{}_{}_Slice.csv'.format(disease, cphase)
        with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader]
        data_infos += rows
        assert data_infos, 'No data loaded.'

        # record case
        record_path = 'Test_{}_{}_Case.csv'.format(disease, cphase)
        with open(os.path.join(dataroot, log_name, record_path), 'r', encoding="UTF-8") as csvfile:
            reader = csv.reader(csvfile)
            rows = [row[0] for row in reader]
        case_infos += rows
        assert case_infos, 'No data loaded.'

    else:
        ValueError('Unknown phase {}'.format(phase))

    data_array = []
    data_info = []

    if debug:
        data_infos = data_infos[0:100]

    for data_idx, info in enumerate(data_infos):

        case_name, slice_name = info[0], info[1]

        data_path = os.path.join(dataroot, 'h5', '{}.h5'.format(case_name))
        img_info = '{}_{}'.format(case_name, slice_name)

        data_dict = read_h5(data_path, case_name, slice_name)

        img_H = data_dict['pixel_array']

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