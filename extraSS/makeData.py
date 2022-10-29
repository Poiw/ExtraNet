import numpy as np
import os
from os.path import join as pjoin
import imageio
import sys
from glob import glob
from tqdm import tqdm

import toolFuncs

#################### Parameters #################################

tar_dir = '/home/Data/disk1/Songyin/Data/seq1'
use_demodulation = True
glossy_demodulation = True

#################################################################


Prefixes = ["MotionVector", "PreTonemapHDRColor", "NoV", "MyStencil", "WorldPosition", "WorldNormal", "Specular", "SceneDepth", "Metallic", "Roughness", "BaseColor"]
High_Prefixes = ["MotionVector", "PreTonemapHDRColor"]

def checkExist(index):


    for prefix in Prefixes:
        if not os.path.exists(pjoin(tar_dir, prefix+'.{:04d}.exr'.format(index))):
            return False

    for prefix in High_Prefixes:
        if not os.path.exists(pjoin(tar_dir, "High+TAA", prefix+'.{:04d}.exr'.format(index))):
            return False

    return True

def load_exr_img(path):
    return imageio.imread(path, "exr")[..., :3]
    


if __name__ == '__main__':

    file_list = glob(pjoin(tar_dir, "BaseColor*"))

    indexes = []
    for info in file_list:
        name = os.path.basename(info)
        indexes.append(int(name.split('.')[-2]))

    Min_Index = np.min(indexes)
    Max_Index = np.max(indexes)

    print("Checking files from {} to {}......".format(Min_Index, Max_Index))

    for i in tqdm(range(Min_Index, Max_Index+1)):
        if not checkExist(i):
            print("File with index {} is missing".format(i))
            exit(0)

    print("Check done. Range from {} to {}.\n".format(Min_Index, Max_Index))


    os.makedirs(pjoin(tar_dir, 'warp'))
    os.makedirs(pjoin(tar_dir, 'occ_warp'))

    if use_demodulation:
        os.makedirs(pjoin(tar_dir, 'demodulation'))

        print('Demodulating...')

        for idx in tqdm(range(Min_Index, Max_Index+1)):
            color = load_exr_img(pjoin(tar_dir, "PreTonemapHDRColor.{:04d}.exr".format(idx)))
            basecolor = load_exr_img(pjoin(tar_dir, "BaseColor.{:04d}.exr".format(idx)))

            if glossy_demodulation:
                specular = load_exr_img(pjoin(tar_dir, "Specular.{:04d}.exr".format(idx)))
                metallic = load_exr_img(pjoin(tar_dir, "Metallic.{:04d}.exr".format(idx)))

                demodulation = color / ( basecolor + specular * 0.08 * (1-metallic) )
            else:
                demodulation = color / basecolor

            demodulation[basecolor == 0] = 0

            imageio.imwrite(pjoin(tar_dir, 'demodulation', "Demodulation.{:04d}.exr".format(idx)), demodulation)


        print("Demodulation done.\n")
            



    prevNormal = load_exr_img(pjoin(tar_dir, "WorldNormal.{:04d}.exr".format(Min_Index)))
    prevStencil = load_exr_img(pjoin(tar_dir, "MyStencil.{:04d}.exr".format(Min_Index)))
    prevPosition = load_exr_img(pjoin(tar_dir, "WorldPosition.{:04d}.exr".format(Min_Index)))
    
    print("Warping ...")

    for idx in tqdm(range(Min_Index+1, Max_Index+1)):

        # Get previous frame
        if use_demodulation:
            prevImg = load_exr_img(pjoin(tar_dir, "demodulation", "Demodulation.{:04d}.exr".format(idx-1)))
        else:
            prevImg = load_exr_img(pjoin(tar_dir, "PreTonemapHDRColor.{:04d}.exr".format(idx-1)))

        motionVector = load_exr_img(pjoin(tar_dir, "MotionVector.{:04d}.exr".format(idx)))
        normal = load_exr_img(pjoin(tar_dir, "WorldNormal.{:04d}.exr".format(idx)))
        stencil = load_exr_img(pjoin(tar_dir, "MyStencil.{:04d}.exr".format(idx)))
        position = load_exr_img(pjoin(tar_dir, "WorldPosition.{:04d}.exr".format(idx)))
        depth = load_exr_img(pjoin(tar_dir, "SceneDepth.{:04d}.exr".format(idx)))
        nov = load_exr_img(pjoin(tar_dir, "NoV.{:04d}.exr".format(idx)))

        warpped_img, hole = toolFuncs.warp_img_with_hole(prevImg, 
                                                        motionVector, 
                                                        [prevNormal, normal], 
                                                        [prevStencil, stencil],
                                                        [prevPosition, position], 
                                                        depth,
                                                        nov)

        warpped_img[hole] = -1
        imageio.imwrite(pjoin(tar_dir, "warp", "Warp.{:04d}.exr".format(idx)), warpped_img)

        occMotionVector = toolFuncs.GetOCCMV(motionVector, depth)
        occWarpped_img = toolFuncs.warp_img(prevImg, occMotionVector)

        imageio.imwrite(pjoin(tar_dir, "occ_warp", "Warp.{:04d}.exr".format(idx)), occWarpped_img)
        imageio.imwrite(pjoin(tar_dir, "occ_warp", "MotionVector.{:04d}.exr".format(idx)), occMotionVector)
        


    print("Warping done.\n")

    print("done.")



