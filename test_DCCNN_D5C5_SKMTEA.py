#!/usr/bin/env python

import time

import theano
import theano.tensor as T

import lasagne
import argparse

from os.path import join

from utils import compressed_sensing as cs

from cascadenet.network.model import build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format

from dataloader.data_loader_SKMTEA import *
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


def iterate_minibatch(data, batch_size, shuffle=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]


def iterate_minibatch_test(data, data_info, batch_size=1, shuffle=False):

    n = len(data)
    assert len(data) == len(data_info)

    assert batch_size == 1
    assert shuffle == False

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size], data_info[i:i+batch_size]


def create_dummy_data(data_path_train, data_path_val, data_path_test, h, w):

    train_data_array, train_data_info = load_images(data_path_train, h, w)
    val_data_array, val_data_info = load_images(data_path_val, h, w)
    test_data_array, test_data_info = load_images(data_path_test, h, w)

    return train_data_array, val_data_array, test_data_array, train_data_info, val_data_info, test_data_info


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
    parser.add_argument('--task_name', type=str,)
    parser.add_argument('--data_path_train', type=str, default="/media/ssd/data_temp/fastMRI/knee/d.1.0.complex/train/PD/h5_image_complex",)
    parser.add_argument('--data_path_val', type=str, default="/media/ssd/data_temp/fastMRI/knee/d.1.0.complex/val/PD/h5_image_complex",)
    parser.add_argument('--data_path_test', type=str, default="/media/ssd/data_temp/fastMRI/knee/d.1.0.complex/test/PD/h5_image_complex",)
    parser.add_argument('--weight_path', type=str, default="/media/NAS01/jiahao/DCCNN/FastMRI", )
    parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--l2', type=float, default=1e-6, help='l2 regularisation')
    parser.add_argument('--undersampling_mask', type=str, default="fMRI_Ran_AF4_CF0.08_PE320", help='Undersampling mask for k-space sampling')
    parser.add_argument('--resolution', type=int, default=320, help='Undersampling mask for k-space sampling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')

    args = parser.parse_args()

    print(theano.config.device)

    # Project config
    undersampling_mask = args.undersampling_mask
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    Nx, Ny = args.resolution, args.resolution
    save_fig = args.savefig
    save_every = 1
    model_name = 'DCCNN_D5C5_SKMTEA_{}'.format(args.undersampling_mask)
    data_path_train = args.data_path_train
    data_path_val = args.data_path_val
    data_path_test = args.data_path_test

    # Configure directory info
    project_root = '/home/jh/Deep-MRI-Reconstruction_py3'
    save_dir = join(project_root, 'results/%s' % model_name)
    weight_path = os.path.join(args.weight_path, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)
    net_config, net,  = build_d5_c5(input_shape)

    # D5-C5 with pre-trained parameters
    with np.load(os.path.join(weight_path, f'{model_name}_epoch_50.npz')) as f:
        param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net, param_values)

    # Compile function
    train_fn, val_fn = compile_fn(net, net_config, args)

    # Create dataset
    train, validate, test, train_info, val_info, test_info = create_dummy_data(data_path_train,
                                                                               data_path_val,
                                                                               data_path_test,
                                                                               Nx,
                                                                               Ny)

    print('Start Training...')

    # Testing
    print('Testing')
    vis = []
    i = 0

    # Load mask
    # mask = load_mask(undersampling_mask)
    # mask = np.repeat(mask[:, np.newaxis], mask.shape[0], axis=1).transpose((1, 0))
    if 'fMRI' in undersampling_mask:
        mask_1d = load_mask(undersampling_mask)
        mask_1d = mask_1d[:, np.newaxis]
        mask = np.repeat(mask_1d, args.image_size, axis=1).transpose((1, 0))[:, :]  # (320, 320)
    else:
        mask = load_mask(undersampling_mask)

    mask = scipy.fftpack.ifftshift(mask)
    mask_bs = mask[np.newaxis, :, :]
    mask_complex = np.repeat(mask_bs, batch_size, axis=0).astype(float)

    for im, im_info in tqdm(iterate_minibatch_test(test, test_info, 1, shuffle=False)):

        t_start = time.time()

        slice_info = im_info[0]
        im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)

        err, pred = val_fn(im_und, mask, k_und, im_gnd)

        for im_i, und_i, pred_i in zip(im, from_lasagne_format(im_und), from_lasagne_format(pred)):

            img_gt_cplx = im[0]
            img_recon_cplx = from_lasagne_format(pred)[0]
            img_zf_cplx = from_lasagne_format(im_und)[0]

            mkdir(os.path.join(save_dir, 'test_epoch_50', 'h5'))
            with h5py.File(os.path.join(save_dir, 'test_epoch_50', 'h5', '{}.h5'.format(slice_info)), "w") as file:
                file.create_group("gt")
                file['gt_cplx'] = img_gt_cplx
                file.create_group("recon")
                file['recon_cplx'] = img_recon_cplx
                file.create_group("zf")
                file['zf_cplx'] = img_zf_cplx
                file.attrs['img_info'] = '{}'.format(slice_info)

            # gt = abs(img_gt_cplx)
            # recon = abs(img_recon_cplx)
            # zf = abs(img_zf_cplx)
            #
            # mkdir(os.path.join(save_dir, 'test_epoch_50', 'png', 'GT'))
            # cv2.imwrite(os.path.join(save_dir, 'test_epoch_50', 'png', 'GT', 'GT_{:04d}.png'.format(i)), gt*255)
            # mkdir(os.path.join(save_dir, 'test_epoch_50', 'png', 'Recon'))
            # cv2.imwrite(os.path.join(save_dir, 'test_epoch_50', 'png', 'Recon', 'Recon_{:04d}.png'.format(i)), recon*255)
            # mkdir(os.path.join(save_dir, 'test_epoch_50', 'png', 'ZF'))
            # cv2.imwrite(os.path.join(save_dir, 'test_epoch_50', 'png', 'ZF', 'ZF_{:04d}.png'.format(i)), zf*255)

        i = i + 1
        t_end = time.time()
        print('Testing Idx: {}; Time Cost: {}'.format(i, (t_end-t_start)))



