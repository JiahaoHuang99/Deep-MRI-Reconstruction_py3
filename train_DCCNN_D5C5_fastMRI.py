#!/usr/bin/env python

import time

import theano
import theano.tensor as T

import lasagne
import argparse

from os.path import join

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet.network.model import build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format

from dataloader.data_loader_fastMRI import *
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


def iterate_minibatch(data, batch_size, shuffle=True, drop_last=True):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    if drop_last:
        n = n - n % batch_size

    for i in range(0, n, batch_size):
        yield data[i:i+batch_size]



def create_dummy_data(data_path_train, data_path_val, data_path_test, h, w):

    train_data_array, train_data_info = load_images(data_path_train, h, w, debug=False)
    val_data_array, val_data_info = load_images(data_path_val, h, w, debug=False)
    test_data_array, test_data_info = load_images(data_path_test, h, w, debug=False)

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
    image_size = args.resolution
    save_fig = args.savefig
    save_every = 1
    model_name = 'DCCNN_D5C5_FastMRI_{}'.format(args.undersampling_mask)
    data_path_train = args.data_path_train
    data_path_val = args.data_path_val
    data_path_test = args.data_path_test

    # Configure directory info
    project_root = '/home/jh/Deep-MRI-Reconstruction_py3'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)
    net_config, net,  = build_d5_c5(input_shape)

    # Load D5-C5 with pretrained params
    epoch_updated = 0

    # with np.load(pretrain_path) as f:
    #     param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
    #     lasagne.layers.set_all_param_values(net, param_values)

    # Compile function
    train_fn, val_fn = compile_fn(net, net_config, args)

    # Create dataset
    train, validate, test, train_info, val_info, test_info = create_dummy_data(data_path_train,
                                                                               data_path_val,
                                                                               data_path_test,
                                                                               Nx,
                                                                               Ny)

    print('Start Training...')
    for epoch in range(num_epoch):

        epoch = epoch + epoch_updated

        t_start = time.time()

        # Training
        print('Training')
        train_err = 0
        train_batches = 0

        # Load mask
        # mask = load_mask(undersampling_mask)
        # mask = np.repeat(mask[:, np.newaxis], mask.shape[0], axis=1).transpose((1, 0))
        if 'fMRI' in undersampling_mask:
            mask_1d = load_mask(undersampling_mask)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, image_size, axis=1).transpose((1, 0))[:, :]  # (320, 320)
        else:
            mask = load_mask(undersampling_mask)

        mask = scipy.fftpack.ifftshift(mask)
        mask_bs = mask[np.newaxis, :, :]
        mask_complex = np.repeat(mask_bs, batch_size, axis=0).astype(float)

        for im in tqdm(iterate_minibatch(train, batch_size, shuffle=True, drop_last=True)):

            im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)  # (BS, 2, 128, 128) # mask (BS, 2, 128, 128)
            err = train_fn(im_und, mask, k_und, im_gnd)[0]
            train_err += err
            train_batches += 1

        train_err /= train_batches
        t_end = time.time()

        # Testing
        print('Testing')
        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        i = 0

        # Load mask
        mask_complex = np.repeat(mask_bs, 1, axis=0).astype(float)
        if (epoch + 1) % 1 == 0:
            for im in tqdm(iterate_minibatch(test, 1, shuffle=False, drop_last=False)):
                im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)

                err, pred = val_fn(im_und, mask, k_und, im_gnd)

                test_err += err
                for im_i, und_i, pred_i in zip(im, from_lasagne_format(im_und), from_lasagne_format(pred)):
                    base_psnr += complex_psnr(im_i, und_i, peak='max')
                    test_psnr += complex_psnr(im_i, pred_i, peak='max')

                    gt = abs(im)[0]
                    recon = abs(from_lasagne_format(pred))[0]
                    zf = abs(from_lasagne_format(im_und))[0]

                    if save_fig and i < 10:
                        mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'GT'))
                        cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'GT', 'GT_{:04d}.png'.format(i)), gt*255)
                        mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'Recon'))
                        cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'Recon', 'Recon_{:04d}.png'.format(i)), recon*255)
                        mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'ZF'))
                        cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch + 1), 'png', 'ZF', 'ZF_{:04d}.png'.format(i)), zf*255)

                    test_batches += 1
                    i = i + 1

            test_err /= test_batches
            base_psnr /= (test_batches * batch_size)
            test_psnr /= (test_batches * batch_size)

        print("Epoch {}/{}".format(epoch + 1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        if (epoch + 1) % 20 == 0:
            print(" test loss:\t\t{:.6f}".format(test_err))
            print(" base PSNR:\t\t{:.6f}".format(base_psnr))
            print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # save the model
        if (epoch + 1) % 5 == 0:
            name = '%s_epoch_%d.npz' % (model_name, epoch + 1)
            np.savez(join(save_dir, name), *lasagne.layers.get_all_param_values(net))
            print('model parameters saved at %s' % join(os.getcwd(), name))

