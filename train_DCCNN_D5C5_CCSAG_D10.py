#!/usr/bin/env python

import time

import theano
import theano.tensor as T

import lasagne
import argparse
import matplotlib.pyplot as plt

from os.path import join

from utils import compressed_sensing as cs
from utils.metric import complex_psnr

from cascadenet.network.model import build_d5_c5
from cascadenet.util.helpers import from_lasagne_format
from cascadenet.util.helpers import to_lasagne_format

from dataloader.data_loader_CC import *
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
    # parser.add_argument('--undersampling_mask', metavar='str', nargs=1,
    #                     default=['G2D30'],
    #                     help='Undersampling mask for k-space sampling')
    # parser.add_argument('--gauss_ivar', metavar='float', nargs=1,
    #                     default=['0.0015'],
    #                     help='Sensitivity for Gaussian Distribution which'
    #                     'decides the undersampling rate of the Cartesian mask')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--savefig', action='store_true', help='Save output images and masks')

    args = parser.parse_args()

    print(theano.config.device)
    # Project config
    mask_name = 'G1D10'
    model_name = 'DCCNN_D5C5_CC_{}'.format(mask_name)

    num_epoch = int(args.num_epoch[0])
    batch_size = int(args.batch_size[0])
    Nx, Ny = 256, 256
    save_fig = args.savefig
    save_every = 1

    # Configure directory info
    project_root = '.'
    save_dir = join(project_root, 'models/%s' % model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Specify network
    input_shape = (batch_size, 2, Nx, Ny)
    net_config, net,  = build_d5_c5(input_shape)

    # # Load D5-C5 with pretrained params
    # net_config, net,  = build_d5_c5(input_shape)
    # D5-C5 with pre-trained parameters
    # with np.load('./models/pretrained/d5_c5.npz') as f:
    #     param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
    #     lasagne.layers.set_all_param_values(net, param_values)

    # Compute acceleration rate
    # dummy_mask = cs.cartesian_mask((10, Nx, Ny), acc, sample_n=8)
    # sample_und_factor = cs.undersampling_rate(dummy_mask)
    # print('Undersampling Rate: {:.2f}'.format(sample_und_factor))

    # Compile function
    train_fn, val_fn = compile_fn(net, net_config, args)

    # Create dataset
    train, validate, test = create_dummy_data()

    print('Start Training...')
    for epoch in range(num_epoch):
        t_start = time.time()

        # Training
        print('Training')
        train_err = 0
        train_batches = 0
        # Load mask
        mask_complex = load_mask(batch_size, mask_name)
        for im in tqdm(iterate_minibatch(train, batch_size, shuffle=True)):
            im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)  # (BS, 2, 128, 128) # mask (BS, 2, 128, 128)
            err = train_fn(im_und, mask, k_und, im_gnd)[0]
            train_err += err
            train_batches += 1

            # if args.debug and train_batches == 20:
            #     break

        # Validation
        # validate_err = 0
        # validate_batches = 0
        # for im in tqdm(iterate_minibatch(validate, batch_size, shuffle=False)):
        #     im_und, k_und, mask, im_gnd = prep_input(im, mask_complex)
        #     err, pred = val_fn(im_und, mask, k_und, im_gnd)
        #     validate_err += err
        #     validate_batches += 1
        #
        #     # if args.debug and validate_batches == 20:
        #     #     break

        # Testing
        print('Testing')
        vis = []
        test_err = 0
        base_psnr = 0
        test_psnr = 0
        test_batches = 0
        i = 0
        # Load mask
        mask_complex = load_mask(1, mask_name)
        if (epoch+1)%5 == 0:
            for im in tqdm(iterate_minibatch(test, 1, shuffle=False)):
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

                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'GT'))
                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'GT'))
                    np.save(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'GT', 'GT_{:04d}.npy'.format(i)), gt)
                    cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'GT', 'GT_{:04d}.png'.format(i)), gt*255)
                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'Recon'))
                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'Recon'))
                    np.save(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'Recon', 'Recon_{:04d}.npy'.format(i)), recon)
                    cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'Recon', 'Recon_{:04d}.png'.format(i)), recon*255)
                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'ZF'))
                    mkdir(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'ZF'))
                    np.save(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'npy', 'ZF', 'ZF_{:04d}.npy'.format(i)), zf)
                    cv2.imwrite(os.path.join(save_dir, 'epoch_{}'.format(epoch), 'png', 'ZF', 'ZF_{:04d}.png'.format(i)), zf*255)

                    test_batches += 1
                    i = i + 1
                    # if save_fig and test_batches % save_every == 0:
                    #     vis.append((im[0],
                    #                 from_lasagne_format(pred)[0],
                    #                 from_lasagne_format(im_und)[0],
                    #                 from_lasagne_format(mask, mask=True)[0]))
                    # if args.debug and test_batches == 20:
                    #     break

        t_end = time.time()

        # train_err /= train_batches
        # validate_err /= validate_batches
        # test_err /= test_batches
        # base_psnr /= (test_batches*batch_size)
        # test_psnr /= (test_batches*batch_size)

        # Then we print the results for this epoch:
        print("Epoch {}/{}".format(epoch+1, num_epoch))
        print(" time: {}s".format(t_end - t_start))
        print(" training loss:\t\t{:.6f}".format(train_err))
        # print(" validation loss:\t{:.6f}".format(validate_err))
        print(" test loss:\t\t{:.6f}".format(test_err))
        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))

        # save the model
        if epoch in [1, 2, num_epoch-1]:
            if save_fig:
                i = 0
                for im_i, pred_i, und_i, mask_i in vis:
                    plt.imsave(join(save_dir, 'im{0}.png'.format(i)),
                               abs(np.concatenate([und_i, pred_i, im_i, im_i - pred_i], 1)), cmap='gray')
                    plt.imsave(join(save_dir, 'mask{0}.png'.format(i)),
                               mask_i, cmap='gray')
                    i += 1

            name = '%s_epoch_%d.npz' % (model_name, epoch)
            np.savez(join(save_dir, name),
                     *lasagne.layers.get_all_param_values(net))
            print('model parameters saved at %s' % join(os.getcwd(), name))

