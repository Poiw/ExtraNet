import torch
import numpy as np
import os
import math
from os.path import join as pjoin
from tqdm import tqdm
from glob import glob
import imageio

import toolFuncs



def depth_preprocess(depth):

    depth[depth > 100] = 0.0
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)


Preprocessing_funcs = {}
Preprocessing_funcs["SceneDepth"] = depth_preprocess



class DataDirInfo():

    def __init__(self, path, name):


        self.data_dir = path
        self.data_name = name
        self.index_list = []

        files = glob(pjoin(self.data_dir, "OCCDebug", "OCCMotionVector*.1.exr"))
        for file_path in files:
            file_name = os.path.basename(file_path)
            idx = int(file_name.split('.')[0][-4:])
            self.index_list.append(idx)

        self.index_list.sort()

        ####################### Supported prefix ###############################
        # PreTonemapHDRColor
        # MotionVector
        # Specular
        # WorldPosition
        # WorldNormal
        # BaseColor
        # MyStencil
        # Roughness
        # Metallic
        # NoV
        # SceneDepth

        # OccMotionVector
        # HighResoTAAPreTonemapHDRColor

        #########################################################################

    def __len__(self):
        return len(self.index_list)

    def getPath(self, Prefix, index, offset=0):

        if Prefix not in [ "PreTonemapHDRColor" ,
                            "MotionVector"      ,
                            "Specular"          ,
                            "WorldPosition"     ,
                            "WorldNormal"       ,
                            "BaseColor"         ,
                            "MyStencil"         ,
                            "Roughness"         ,
                            "Metallic"          ,
                            "NoV"               ,
                            "SceneDepth"        ,
                            "OccMotionVector"   ,
                            "HighResoTAAPreTonemapHDRColor"]:
            raise NotImplementedError("The prefix type is unknown.")

        if Prefix == "OccMotionVector":
            return pjoin(self.data_dir, "OCCDebug", "OCCMotionVector{:04d}.1.exr".format(self.index_list[index] - offset))
        elif Prefix == "HighResoTAAPreTonemapHDRColor":
            return pjoin(self.data_dir, "HighResoTAA", "{}PreTonemapHDRColor.{}.exr".format(self.data_name, self.index_list[index] - offset))
        else:
            return pjoin(self.data_dir, "{}{}.{}.exr".format(self.data_name, Prefix, self.index_list[index] - offset))

    def getChannel(self, Prefix):

        if Prefix not in [ "PreTonemapHDRColor" ,
                            "MotionVector"      ,
                            "Specular"          ,
                            "WorldPosition"     ,
                            "WorldNormal"       ,
                            "BaseColor"         ,
                            "MyStencil"         ,
                            "Roughness"         ,
                            "Metallic"          ,
                            "NoV"               ,
                            "SceneDepth"        ,
                            "OccMotionVector"   ,
                            "HighResoTAAPreTonemapHDRColor"]:
            raise NotImplementedError("The prefix type is unknown.")

        if Prefix in ["Metallic", "Roughness", "Specular", "MyStencil", "NoV", "SceneDepth"]:
            return 1
        else:
            return 3



class extraSS_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, scene_name, used_keys=[]):

        # For warpped or occ_warpped frames, please use the key name "warp_key" or "occ-warp_key"

        self.data_info = DataDirInfo(data_dir, scene_name)
        self.used_keys = used_keys


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):

        # Todo: data augmentation

        data = {}

        for key in self.used_keys:
            if "warp" in key:
                
                unwarpped_key = key.split('_')[-1]

                channel = self.data_info.getChannel(unwarpped_key)

                unwarpped_img = imageio.imread(self.data_info.getPath(unwarpped_key, index, 1))[..., :channel]


                if "occ" in key:
                    motion_vector = imageio.imread(self.data_info.getPath("OccMotionVector", index))[..., :2]
                else:
                    motion_vector = imageio.imread(self.data_info.getPath('MotionVector', index))[..., :2]

                data[key] = toolFuncs.warp_img(unwarpped_img, motion_vector)


            else:
                channel = self.data_info.getChannel(key)
                data[key] = imageio.imread(self.data_info.get_path(key, index))[..., :channel]

        for preprocess_key in Preprocessing_funcs:
            if preprocess_key in data:
                data[preprocess_key] = Preprocessing_funcs[preprocess_key](data[preprocess_key])


        return data

def extraSS_Dataset_syndata(batch):

    data = {}

    for key in batch[0]:
        data[key] = [torch.FloatTensor(batch[0][key].transpose([2, 0, 1]))]

    for i in range(1, len(batch)):
        for key in batch[i]:
            data[key].append(torch.FloatTensor(batch[i][key].transpose([2, 0, 1])))

    for key in data:
        data[key] = torch.stack(data[key], dim=0)

    return data
        


