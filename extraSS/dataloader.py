import torch
import numpy as np
import os
import math
from os.path import join as pjoin
from tqdm import tqdm
from glob import glob
import imageio

import toolFuncs
from scipy import interpolate


def depth_preprocess(depth):

    depth[depth > 100] = 0.0
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    return depth


Preprocessing_funcs = {}
Preprocessing_funcs["SceneDepth"] = depth_preprocess

# This one is saved to the data directory
# Valid_Data_prefix = [ "PreTonemapHDRColor" ,
#                             "demodulatePreTonemapHDRColor",
#                             "MotionVector"      ,
#                             "Specular"          ,
#                             "WorldPosition"     ,
#                             "WorldNormal"       ,
#                             "BaseColor"         ,
#                             "MyStencil"         ,
#                             "Roughness"         ,
#                             "Metallic"          ,
#                             "NoV"               ,
#                             "SceneDepth"        ,
#                             "OccMotionVector"   ,
#                             "HighResoTAAPreTonemapHDRColor",
#                             "warp_demodulatePreTonemapHDRColor",
#                             "occ-warp_demodulatePreTonemapHDRColor"]

# These are old version dataloader
# class DataDirInfo():

#     def __init__(self, path, name):


#         self.data_dir = path
#         self.data_name = name
#         self.index_list = []

#         files = glob(pjoin(self.data_dir, "OCCDebug", "OCCMotionVector*.1.exr"))
#         for file_path in files:
#             file_name = os.path.basename(file_path)
#             idx = int(file_name.split('.')[0][-4:])
#             self.index_list.append(idx)

#         self.index_list.sort()

#         ####################### Supported prefix ###############################
#         # PreTonemapHDRColor
#         # MotionVector
#         # Specular
#         # WorldPosition
#         # WorldNormal
#         # BaseColor
#         # MyStencil
#         # Roughness
#         # Metallic
#         # NoV
#         # SceneDepth

#         # OccMotionVector
#         # HighResoTAAPreTonemapHDRColor

#         #########################################################################

#     def __len__(self):
#         return len(self.index_list)

#     def getPath(self, Prefix, index, offset=0):

#         if Prefix not in Valid_Data_prefix:
#             raise NotImplementedError("The prefix type is unknown.")

#         if Prefix == "OccMotionVector":
#             return pjoin(self.data_dir, "OCCDebug", "OCCMotionVector{:04d}.1.exr".format(self.index_list[index] - offset))
#         elif Prefix == "HighResoTAAPreTonemapHDRColor":
#             return pjoin(self.data_dir, "HighResoTAA", "{}PreTonemapHDRColor.{:04d}.exr".format(self.data_name, self.index_list[index] - offset))
#         elif Prefix == "demodulatePreTonemapHDRColor":
#             return pjoin(self.data_dir, "demodulate", "{}PreTonemapHDRColor.{:04d}.exr".format(self.data_name, self.index_list[index] - offset))
#         else:
#             return pjoin(self.data_dir, "{}{}.{:04d}.exr".format(self.data_name, Prefix, self.index_list[index] - offset))

#     def getChannel(self, Prefix):

#         if Prefix not in Valid_Data_prefix:
#             raise NotImplementedError("The prefix type is unknown.")

#         if Prefix in ["Metallic", "Roughness", "Specular", "MyStencil", "NoV", "SceneDepth"]:
#             return 1
#         else:
#             return 3



# class extraSS_Dataset(torch.utils.data.Dataset):

#     def __init__(self, data_dir, scene_name, used_keys=[]):

#         # For warpped or occ_warpped frames, please use the key name "warp_key" or "occ-warp_key"

#         self.data_info = DataDirInfo(data_dir, scene_name)
#         self.used_keys = used_keys


#     def __len__(self):
#         return len(self.data_info)

#     def __getitem__(self, index):

#         # Todo: data augmentation

#         data = {}

#         for key in self.used_keys:
#             if "warp" in key:
                
#                 unwarpped_key = key.split('_')[-1]

#                 channel = self.data_info.getChannel(unwarpped_key)

#                 unwarpped_img = imageio.imread(self.data_info.getPath(unwarpped_key, index, 1), "exr")[..., :channel]


#                 if "occ" in key:
#                     motion_vector = imageio.imread(self.data_info.getPath("OccMotionVector", index), "exr")[..., :3]
#                     # Use this only when data is not fixed
#                     motion_vector[..., 0] = motion_vector[..., 2]
#                     motion_vector = motion_vector[..., :2]
#                 else:
#                     motion_vector = imageio.imread(self.data_info.getPath('MotionVector', index), "exr")[..., :2]

#                 # Reference: https://stackoverflow.com/questions/41879104/upsample-and-interpolate-a-numpy-array
#                 if motion_vector.shape[0] != unwarpped_img.shape[0] or motion_vector.shape[1] != unwarpped_img.shape[1]:

#                     ratio_x = (unwarpped_img.shape[1] / motion_vector.shape[1])
#                     ratio_y = (unwarpped_img.shape[0] / motion_vector.shape[0])

#                     motion_vector[..., 1] = motion_vector[..., 1] * ratio_y
#                     motion_vector[..., 0] = motion_vector[..., 0] * ratio_x

#                     x = np.array(range(motion_vector.shape[1]))
#                     y = np.array(range(motion_vector.shape[0]))

#                     mv_x_f = interpolate.interp2d(x, y, motion_vector[..., 0])
#                     mv_y_f = interpolate.interp2d(x, y, motion_vector[..., 1])

#                     xnew = (np.array(range(unwarpped_img.shape[1])) - 0.5) / ratio_x
#                     ynew = (np.array(range(unwarpped_img.shape[0])) - 0.5) / ratio_y

#                     mv_x = mv_x_f(xnew, ynew)
#                     mv_y = mv_y_f(xnew, ynew)

#                     motion_vector = np.stack([mv_x, mv_y], axis=2)

#                 data[key] = toolFuncs.warp_img(unwarpped_img, motion_vector)


#             elif "previous" in key:
#                 original_key = key.split('_')[-1]

#                 channel = self.data_info.getChannel(original_key)

#                 previous_img = imageio.imread(self.data_info.getPath(original_key, index, 1), "exr")[..., :channel]

#                 data[key] = previous_img


#             else:
#                 channel = self.data_info.getChannel(key)
#                 data[key] = imageio.imread(self.data_info.getPath(key, index), "exr")[..., :channel]

#         for preprocess_key in Preprocessing_funcs:
#             if preprocess_key in data:
#                 data[preprocess_key] = Preprocessing_funcs[preprocess_key](data[preprocess_key])


#         return data



class DataDirInfo():

    def __init__(self, path):


        self.data_dir = path
        self.index_list = []
        self.Valid_Data_prefix = []

        with open(pjoin(path, 'valid_data_prefix.log')) as f:
            buf = f.readlines()

        for line in buf:
            self.Valid_Data_prefix.append(line.rstrip('\n').rstrip(' '))

        files = glob(pjoin(self.data_dir, "warp", "Warp.*.exr"))
        for file_path in files:
            file_name = os.path.basename(file_path)
            idx = int(file_name.split('.')[-2])
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

        if Prefix not in self.Valid_Data_prefix:
            raise NotImplementedError("The prefix type is unknown.")

        query_index = self.index_list[index] - offset

        if Prefix == "OccMotionVector":
            return pjoin(self.data_dir, "occ_warp", "MotionVector.{:04d}.exr".format(query_index))
        elif Prefix == "warp_demodulatePreTonemapHDRColor":
            return pjoin(self.data_dir, "warp", "Warp.{:04d}.exr".format(query_index))
        elif Prefix == "occ-warp_demodulatePreTonemapHDRColor":
            return pjoin(self.data_dir, "occ_warp", "Warp.{:04d}.exr".format(query_index))
        elif Prefix == "HighResoTAAPreTonemapHDRColor":
            return pjoin(self.data_dir, "High+TAA", "PreTonemapHDRColor.{:04d}.exr".format(query_index))
        elif Prefix == "demodulatePreTonemapHDRColor":
            return pjoin(self.data_dir, "demodulation", "Demodulation.{:04d}.exr".format(query_index))
        else:
            return pjoin(self.data_dir, "{}.{:04d}.exr".format(Prefix, query_index))

    def getChannel(self, Prefix):

        if Prefix not in self.Valid_Data_prefix:
            raise NotImplementedError("The prefix type is unknown.")

        if Prefix in ["Metallic", "Roughness", "Specular", "MyStencil", "NoV", "SceneDepth"]:
            return 1
        else:
            return 3



class extraSS_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dir, used_keys=[]):

        # For warpped or occ_warpped frames, please use the key name "warp_key" or "occ-warp_key"

        self.data_info = DataDirInfo(data_dir)
        self.used_keys = used_keys


    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):

        # Todo: data augmentation

        data = {}

        for key in self.used_keys:
            if "warp" in key:

                if key in self.data_info.Valid_Data_prefix:
                    channel = self.data_info.getChannel(key)
                    data[key] = imageio.imread(self.data_info.getPath(key, index), "exr")[..., :channel]
                
                else:
                    unwarpped_key = key.split('_')[-1]

                    channel = self.data_info.getChannel(unwarpped_key)

                    unwarpped_img = imageio.imread(self.data_info.getPath(unwarpped_key, index, 1), "exr")[..., :channel]


                    if "occ" in key:
                        motion_vector = imageio.imread(self.data_info.getPath("OccMotionVector", index), "exr")[..., :3]
                        # Use this only when data is not fixed
                        motion_vector[..., 0] = motion_vector[..., 2]
                        motion_vector = motion_vector[..., :2]
                    else:
                        motion_vector = imageio.imread(self.data_info.getPath('MotionVector', index), "exr")[..., :2]

                    # Reference: https://stackoverflow.com/questions/41879104/upsample-and-interpolate-a-numpy-array
                    if motion_vector.shape[0] != unwarpped_img.shape[0] or motion_vector.shape[1] != unwarpped_img.shape[1]:

                        ratio_x = (unwarpped_img.shape[1] / motion_vector.shape[1])
                        ratio_y = (unwarpped_img.shape[0] / motion_vector.shape[0])

                        motion_vector[..., 1] = motion_vector[..., 1] * ratio_y
                        motion_vector[..., 0] = motion_vector[..., 0] * ratio_x

                        x = np.array(range(motion_vector.shape[1]))
                        y = np.array(range(motion_vector.shape[0]))

                        mv_x_f = interpolate.interp2d(x, y, motion_vector[..., 0])
                        mv_y_f = interpolate.interp2d(x, y, motion_vector[..., 1])

                        xnew = (np.array(range(unwarpped_img.shape[1])) - 0.5) / ratio_x
                        ynew = (np.array(range(unwarpped_img.shape[0])) - 0.5) / ratio_y

                        mv_x = mv_x_f(xnew, ynew)
                        mv_y = mv_y_f(xnew, ynew)

                        motion_vector = np.stack([mv_x, mv_y], axis=2)

                    data[key] = toolFuncs.warp_img(unwarpped_img, motion_vector)


            else:
                channel = self.data_info.getChannel(key)
                data[key] = imageio.imread(self.data_info.getPath(key, index), "exr")[..., :channel]

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
        


