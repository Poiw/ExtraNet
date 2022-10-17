import numpy as np
import os
from os.path import join as pjoin
from glob import glob
import cv2
import imageio
from PIL import Image
from tqdm import tqdm


target_dir = 'D:\Projects\ExtraNet\Test_Res\gt_last_frame_ss-extrapolation'
target_prefix = ['low_pred', 'high_pred', 'HighResoTAAPreTonemapHDRColor']


def gamma_correct(x, p=1/2.2):
    return x ** p

if __name__ == '__main__':

    video_dir = pjoin(target_dir, 'make_videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    names = glob(pjoin(target_dir, target_prefix[0]+'_*'))

    indexes = []
    for name in names:
        indexes.append(int(name.rstrip('.exr').split('_')[-1]))

    for idx in range(500, 700):

        low = imageio.imread(pjoin(target_dir, 'low_pred_{}.exr'.format(idx)), "exr")[..., :3]
        high = imageio.imread(pjoin(target_dir, 'high_pred_{}.exr'.format(idx)), "exr")[..., :3]
        gt = imageio.imread(pjoin(target_dir, 'HighResoTAAPreTonemapHDRColor_{}.exr'.format(idx)), "exr")[..., :3]

        low = cv2.resize(low, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        res = np.concatenate([low, high, gt], axis=1)
        res = gamma_correct(res)
        res = np.clip(res, 0., 1.)

        Image.fromarray((res * 255).astype(np.uint8)).save(pjoin(video_dir, '{:04d}.png'.format(idx)))
