#!/usr/bin/env python

import os
import time

import cv2
import numpy as np

import theano
import theano.tensor as T

import lasagne
import argparse
import matplotlib.pyplot as plt

from os.path import join
from scipy.io import loadmat

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet.network.model import build_d2_c2, build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format

from data_loader_CC import *
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
    # mask = cs.cartesian_mask(im.shape, acc, sample_n=8)
    # im_und, k_und = cs.undersample(im, mask, centred=False, norm='ortho')
    im_gnd_l = to_lasagne_format(im)
    im_und_l = to_lasagne_format(im_und)
    k_und_l = to_lasagne_format(k_und)
    mask_l = to_lasagne_format(mask, mask=True)

    return im_und_l, k_und_l, mask_l, im_gnd_l


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def create_dummy_data():


    train_data_array = load_images('/media/ssd/data_temp/CC/db_train', 47, 100)
    val_data_array = load_images('/media/ssd/data_temp/CC/db_train', 20, 100)

    return train_data_array, val_data_array, val_data_array


def compile_fn(network, net_config, args):
    """
    Create Training function and validation function
    """
    # Hyper-parameters
    base_lr = float(args.lr[0])
    l2 = float(args.l2[0])

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
    parser.add_argument('--num_epoch', metavar='int', nargs=1, default=['10'],
                        help='number of epochs')
    parser.add_argument('--batch_size', metavar='int', nargs=1, default=['10'],
                        help='batch size')
    parser.add_argument('--lr', metavar='float', nargs=1,
                        default=['0.001'], help='initial learning rate')
    parser.add_argument('--l2', metavar='float', nargs=1,
                        default=['1e-6'], help='l2 regularisation')
    # parser.add_argument('--acceleration_factor', metavar='float', nargs=1,
    #                     default=['4.0'],
    #                     help='Acceleration factor for k-space sampling')
    parser.add_argument('--undersampling_mask', metavar='str', nargs=1,
                        default=['G1D30'],
                        help='Undersampling mask for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')

    args = parser.parse_args()

    print(theano.config.device)
    # Project config
    model_name = 'DCCNN_D5C5_CC_G1D30'
    mask_name = str(args.undersampling_mask[0])  # undersampling rate
    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny = 256, 256

    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    mkdir(os.path.join(save_dir))

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)

    # Load D5-C5 with pretrained params
    net_config, net,  = build_d5_c5(input_shape)
    # D5-C5 with pre-trained parameters
    with np.load(os.path.join(save_dir, f'{model_name}_epoch_9.npz')) as f:
        param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)

    # Compile function
    train_fn, val_fn = compile_fn(net, net_config, args)

    # Create dataset
    train, validate, test = create_dummy_data()

    print('Start Training...')

    # Testing
    print('Testing')
    vis = []
    test_err = 0
    base_psnr = 0
    test_psnr = 0
    i = 0
    # Load mask
    mask_complex = load_mask(1, mask_name)

    for im in tqdm(iterate_minibatch(test, 1, shuffle=False)):
        t_start = time.time()
        im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)

        err, pred = val_fn(im_und, mask, k_und, im_gnd)

        test_err += err
        for im_i, und_i, pred_i in zip(im, from_lasagne_format(im_und), from_lasagne_format(pred)):
            base_psnr += complex_psnr(im_i, und_i, peak='max')
            test_psnr += complex_psnr(im_i, pred_i, peak='max')
            # if (epoch+1) % 5 == 0:
            gt = abs(im)[0]
            recon = abs(from_lasagne_format(pred))[0]
            zf = abs(from_lasagne_format(im_und))[0]

            mkdir(os.path.join(save_dir, 'test_epoch_9', 'npy', 'GT'))
            mkdir(os.path.join(save_dir, 'test_epoch_9', 'png', 'GT'))
            np.save(os.path.join(save_dir, 'test_epoch_9', 'npy', 'GT', 'GT_{:04d}.npy'.format(i)), gt)
            cv2.imwrite(os.path.join(save_dir, 'test_epoch_9', 'png', 'GT', 'GT_{:04d}.png'.format(i)), gt*255)
            mkdir(os.path.join(save_dir, 'test_epoch_9', 'npy', 'Recon'))
            mkdir(os.path.join(save_dir, 'test_epoch_9', 'png', 'Recon'))
            np.save(os.path.join(save_dir, 'test_epoch_9', 'npy', 'Recon', 'Recon_{:04d}.npy'.format(i)), recon)
            cv2.imwrite(os.path.join(save_dir, 'test_epoch_9', 'png', 'Recon', 'Recon_{:04d}.png'.format(i)), recon*255)
            mkdir(os.path.join(save_dir, 'test_epoch_9', 'npy', 'ZF'))
            mkdir(os.path.join(save_dir, 'test_epoch_9', 'png', 'ZF'))
            np.save(os.path.join(save_dir, 'test_epoch_9', 'npy', 'ZF', 'ZF_{:04d}.npy'.format(i)), zf)
            cv2.imwrite(os.path.join(save_dir, 'test_epoch_9', 'png', 'ZF', 'ZF_{:04d}.png'.format(i)), zf*255)
        i = i + 1
        t_end = time.time()
        print('Testing Idx: {}; Time Cost: {:.3f}'.format(i, (t_end-t_start)))



