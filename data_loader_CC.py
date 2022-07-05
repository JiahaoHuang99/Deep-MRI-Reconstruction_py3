
import os
from pickle import load, dump
import cv2
import gc
import numpy as np
import scipy.io as scio

def load_images(path, case_num, slice_num):

    imgss = []
    for i in range(case_num):
        # print('[{}/{}] case processing'.format(i + 1, case_num))
        imgs = []
        for j in range(slice_num):
            # print('[{}/{}] slice processing'.format(j + 1, slice_num))
            # load GT
            file_name_gt = 'imgGT_{}_{}.npy'.format(i + 1, j + 1)
            file_dir_gt = os.path.join(path, file_name_gt)
            gt = np.load(file_dir_gt).astype(np.float32)
            # scio.savemat(os.path.join(path, 'imgGT_{}_{}.mat'.format(i + 1, j + 1)), {'im_ori': gt})
            gt = np.reshape(gt, (256, 256))

            # 0 ~ 255
            gt = (gt - gt.min()) / (gt.max() - gt.min())

            img = gt.astype(complex)

            imgs.append(img)
        imgss.append(imgs)

    img_array = np.reshape(np.array(imgss), (case_num * slice_num, 256, 256))
    del imgss, imgs, img
    gc.collect()

    return img_array


def save_img(imgs, savedir):

    # for sample
    img_weight = imgs[0, :, :, 0:1]
    sm = imgs[0, :, :, 1:imgs.shape[3]]

    img_weight = (img_weight + 1) * 127.5
    img = img_weight * sm
    sm = sm * 255

    img_weight = img_weight.astype(np.uint8)
    cv2.imwrite(os.path.join(savedir, 'img_weight.png'), img_weight)

    img = img.astype(np.uint8)
    for i in range(imgs.shape[3]-1):
         cv2.imwrite(os.path.join(savedir, 'img_{}.png'.format(i)), img[:,:,i:i+1])

    sm = sm.astype(np.uint8)
    for i in range(imgs.shape[3]-1):
         cv2.imwrite(os.path.join(savedir, 'sm_{}.png'.format(i)), sm[:,:,i:i+1])


if __name__ == "__main__":

    train = load_images(path='./data/PI/db_train', case_num=1, slice_num=1, isSM=True)
    save_img(train, savedir='./data/PI/sample')
