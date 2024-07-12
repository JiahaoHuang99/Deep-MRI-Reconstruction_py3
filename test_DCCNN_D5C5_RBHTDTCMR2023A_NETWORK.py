#!/usr/bin/env python

import time

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import count_params
import argparse

from os.path import join

from utils import compressed_sensing as cs

from cascadenet.network.model import build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format

from dataloader.data_loader_RBHTDTCMR2023A import *
from mask_loader import *

from tqdm import tqdm

def prep_input(im, mask):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """

    im_und, k_und = cs.undersample_kspace(im, mask)
    im_gnd_l = to_lasagne_format(im)
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True, drop_last=False):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    if drop_last:
        n = n - n % batch_size

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def iterate_minibatch_test(data, data_info, batch_size=1, shuffle=False, drop_last=False):

    n = len(data)
    assert len(data) == len(data_info)

    assert batch_size == 1
    assert shuffle == False
    assert drop_last == False

    if drop_last:
        n = n - n % batch_size

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size], data_info[i:i+batch_size]


def create_dummy_data(data_path, h, w, phase='train', disease='', cphase='', debug=False):

    data, data_info = load_images(data_path, h, w, phase, disease, cphase, debug=debug)

    return data, data_info


def compile_fn(network, net_config, args):
    """
    Create Training function and validation function
    """
    # Hyper-parameters
    base_lr = float(args.lr)
    l2 = float(args.l2)

    # Theano variables
    input_var = net_config['input'].input_var
    mask_var = net_config['mask'].input_var
    kspace_var = net_config['kspace_input'].input_var
    target_var = T.tensor4('targets')

    # Objective
    pred = lasagne.layers.get_output(network)
    # complex valued signal has 2 channels, which counts as 1.
    loss_sq = lasagne.objectives.squared_error(target_var, pred).mean() * 2
    if l2:
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss = loss_sq + l2_penalty * l2

    update_rule = lasagne.updates.adam
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = update_rule(loss, params, learning_rate=base_lr)

    print(' Compiling ... ')
    t_start = time.time()
    train_fn = theano.function([input_var, mask_var, kspace_var, target_var],
                               [loss], updates=updates,
                               on_unused_input='ignore')

    val_fn = theano.function([input_var, mask_var, kspace_var, target_var],
                             [loss, pred],
                             on_unused_input='ignore')
    t_end = time.time()
    print(' ... Done, took %.4f s' % (t_end - t_start))

    return train_fn, val_fn


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default="debug")
    parser.add_argument('--data_path', type=str, default="/media/ssd/data_temp/RBHT/DT_CMR_data/RBHT_DTCMR_2023A/d.1.0/",)
    parser.add_argument('--disease', type=str, default='AllDisease', help='AllDisease or HEALTHY')
    parser.add_argument('--cphase', type=str, default='diastole', help='diastole or systole')
    parser.add_argument('--weight_path', type=str, default="debug", )
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularisation')
    parser.add_argument('--undersampling_mask', type=str, default="fMRI_Reg_AF4_CF0.08_PE48", help='Undersampling mask for k-space sampling')
    parser.add_argument('--resolution_h', type=int, default=256, help='Undersampling mask for k-space sampling')
    parser.add_argument('--resolution_w', type=int, default=96, help='Undersampling mask for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')

    args = parser.parse_args()
    print(theano.config.device)
    # if use CPU, add "THEANO_FLAGS='device=cpu'" before the command

    # Project config
    undersampling_mask = args.undersampling_mask
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    Nx, Ny = args.resolution_h, args.resolution_w
    save_fig = args.savefig
    save_every = 1
    data_path = args.data_path
    disease = args.disease
    cphase = args.cphase

    model_name = 'DCCNN_D5C5_RBHTDTCMR2023A_{}_{}_{}'.format(undersampling_mask, disease, cphase)

    if disease == 'MI':
        model_name_for_weight = 'DCCNN_D5C5_RBHTDTCMR2023A_{}_{}_{}'.format(undersampling_mask, 'AllDisease', cphase)
    else:
        model_name_for_weight = model_name

    # Configure directory info
    project_root = '/home/jh/Deep-MRI-Reconstruction_py3'
    save_dir = join(project_root, 'results/%s' % model_name)

    weight_path = os.path.join(args.weight_path, model_name_for_weight)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)
    net_config, net,  = build_d5_c5(input_shape)

    # Calculate #PARAMs
    params = count_params(net)
    print('{} M'.format(params / 1e6))

    # Compile function
    train_fn, val_fn = compile_fn(net, net_config, args)

    # Create dataset
    # train, train_info = create_dummy_data(data_path, Nx, Ny, phase='train', disease=disease, cphase=cphase, debug=False)
    # validate, val_info = create_dummy_data(data_path, Nx, Ny, phase='val', disease=disease, cphase=cphase, debug=False)
    test, test_info = create_dummy_data(data_path, Nx, Ny, phase='test', disease=disease, cphase=cphase, debug=False)

    print('Start Training...')

    # Testing
    print('Testing')
    vis = []
    i = 0

    # Load mask
    mask_1d = load_mask(undersampling_mask)
    mask_1d = mask_1d[:, np.newaxis]
    mask = np.repeat(mask_1d, 128, axis=1).transpose((1, 0))
    mask = np.pad(mask, ((64, 64), (24, 24)), mode='constant')
    mask = scipy.fftpack.ifftshift(mask)
    mask_bs = mask[np.newaxis, :, :]
    mask_complex = np.repeat(mask_bs, batch_size, axis=0).astype(float)

    t_list = []
    i = 0
    for im, im_info in tqdm(iterate_minibatch_test(test, test_info, 1, shuffle=False)):
        i = i + 1
        if i > 30:
            break

        t_start = time.time()

        case_name, slice_name = im_info[0]
        im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)

        t1 = time.time()
        err, pred = val_fn(im_und, mask, k_und, im_gnd)
        t2 = time.time()

        t_used = t2 - t1
        t_list.append(t_used)

        i = i + 1
        t_end = time.time()
        # print('Testing Idx: {}; Time Cost: {}'.format(i, (t_end-t_start)))

    print(t_list)
    t_list = t_list[-10:]

    t_avg = np.average(t_list)
    t_std = np.std(t_list)

    print('Time used (AVG): {}s'.format(round(t_avg, 3)))
    print('Time used (STD): {}s'.format(round(t_std, 3)))

