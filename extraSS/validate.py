
from genericpath import exists
from pyexpat import model
import sys
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter1
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader
import os
import shutil
import numpy as np
from tqdm import tqdm
import argparse
from os.path import join as pjoin

import ExtraNet
import dataloader
import Losses
import toolFuncs
import utils

import config

import time

parser = argparse.ArgumentParser(description='train.py') #
parser.add_argument('--data_dir', type=str, default='E:\\Data\\Bunker\\1', help='data dir')
parser.add_argument('--data_name', type=str, default='Bunker', help='data name')
parser.add_argument('--output_dir', type=str, default='', help='output dir')
parser.add_argument('--model_path', type=str, default='', help='model_path')

parser.add_argument('--input_channels', type=int, default=13, help="input channel number")
parser.add_argument('--output_channels', type=int, default=3, help="output channel number")


args = parser.parse_args()

cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

use_last_frame = False

def train():

    output_dir = args.output_dir
    os.mkdir(output_dir)
    

    ######################### Network Archtecture #############################
    if config.network_type == "ExtraNet_noHistory":
        model = ExtraNet.ExtraNet_noHistory(args.input_channels, args.output_channels)
    elif config.network_type == "ExtraNet_demodulate_noHistory_SS":
        model = ExtraNet.ExtraNet_demodulate_noHistory_SS(args.input_channels, args.output_channels)
    elif config.network_type == "ExtraNet_demodulate_noHistory_SS_blend":
        model = ExtraNet.ExtraNet_demodulate_noHistory_SS_blend(config.input_channels, config.output_channels)
    else:
        raise NotImplementedError
    model.cuda()
    ###########################################################################


    # DataLoader
    valList = dataloader.extraSS_Dataset(args.data_dir, args.data_name, config.Dataloader_Keys)
    valLoader = DataLoader(valList,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                drop_last=False,
                                collate_fn=dataloader.extraSS_Dataset_syndata)


    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])

    last_output = None

    with torch.no_grad():
        for index, data in enumerate(tqdm(valLoader)):

            if index % 2 == 0:
                ss_only_flag = True
            else:
                ss_only_flag = False
            input, extras, _ = toolFuncs.Process_Input(data, type=config.dataType, ss_only_flag=ss_only_flag)
            for key in extras:
                extras[key] = extras[key].cuda()

            if last_output is not None and use_last_frame:
                input["high"] = last_output.cuda()

            pred = model(input)

            data = toolFuncs.Postprocess(data, pred, type=config.dataType)

            last_output = pred["high"].detach().cpu()

            for key in data:
                img = utils.tensorToNumpy(data[key][0])
                utils.save_exr(img, pjoin(output_dir, "{}_{}.exr".format(key, index)))


if __name__ == '__main__':
    train()
    print('done')