
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
parser.add_argument('--data_dir', type=str, default='/home/songyin/Data/disk1/Songyin/Data/seq1', help='data dir')
# parser.add_argument('--data_name', type=str, default='Bunker', help='data name')
parser.add_argument('--log_dir', type=str, default='/home/songyin/Data/disk1/Songyin/Data/log', help='saving dir')
parser.add_argument('--info', type=str, default='', help='short description')

parser.add_argument('--num_works', type=int, default=8, help='data loading threads')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bz', type=int, default=2, help='batch size')

parser.add_argument('--vis_step', type=int, default=5, help="add to tensorboard frequency")

parser.add_argument('--saving_epoch', type=int, default=10, help='saving steps')
parser.add_argument('--total_epoch', type=int, default=200, help='total steps')


args = parser.parse_args()

cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())



def train():

    saving_dir = pjoin(args.log_dir, cur_time + '__' + args.info)
    model_saving_path = pjoin(saving_dir, 'models')
    board_log_path = pjoin(saving_dir, 'board')
    predictions_path = pjoin(saving_dir, 'preds')

    os.mkdir(saving_dir)
    os.mkdir(model_saving_path)
    os.mkdir(board_log_path)
    os.mkdir(predictions_path)
    with open(pjoin(saving_dir, 'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
    shutil.copyfile("config.py", pjoin("{}".format(saving_dir), "config.py"))
        
    ##################################### Tensorboard Inforamtion ###############################
    writer = SummaryWriter(board_log_path)
    writer_iter = 0
    #############################################################################################

    ######################### Network Archtecture #############################
    if config.network_type == "ExtraNet_noHistory":
        model = ExtraNet.ExtraNet_noHistory(config.input_channels, config.output_channels)
    elif config.network_type == "ExtraNet_demodulate_noHistory_SS":
        model = ExtraNet.ExtraNet_demodulate_noHistory_SS(config.input_channels, config.output_channels)
    elif config.network_type == "ExtraNet_demodulate_noHistory_SS_blend":
        model = ExtraNet.ExtraNet_demodulate_noHistory_SS_blend(config.input_channels, config.output_channels)
    else:
        raise NotImplementedError
    model.cuda()
    ###########################################################################

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


    # DataLoader
    trainList = dataloader.extraSS_Dataset(args.data_dir, config.Dataloader_Keys)
    trainLoader = DataLoader(trainList,
                                batch_size=args.bz,
                                shuffle=True,
                                num_workers=args.num_works,
                                drop_last=True,
                                collate_fn=dataloader.extraSS_Dataset_syndata)

    valList = dataloader.extraSS_Dataset(args.data_dir, config.Dataloader_Keys)
    valLoader = DataLoader(valList,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                collate_fn=dataloader.extraSS_Dataset_syndata)


    # Loss Functions
    if config.loss_func_name == "mLoss":
        criterion = Losses.mLoss()
    elif config.loss_func_name == "Multireso_mLoss":
        criterion = Losses.Multireso_mLoss()
    else:
        raise NotImplementedError

    for e in tqdm(range(args.total_epoch)):

        Loss = 0
        Low_l1 = 0
        High_l1 = 0

        for data in tqdm(trainLoader):

            writer_iter += 1

            input, extras, gt = toolFuncs.Process_Input(data, type=config.dataType)
            for key in extras:
                extras[key] = extras[key].cuda()

            pred = model(input)

            loss = criterion(pred, extras, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                low_l1, high_l1 = toolFuncs.getL1(pred, data, type=config.dataType)

            if writer_iter % args.vis_step == 0:
                writer.add_scalar('train/step_loss', loss.item(), writer_iter)


            Loss += loss.item()
            Low_l1 += low_l1.item()
            High_l1 += high_l1.item()


        writer.add_scalar('train/Loss', Loss / len(trainLoader), writer_iter)
        writer.add_scalar('train/Low_L1', Low_l1 / len(trainLoader), writer_iter)
        writer.add_scalar('train/High_L1', High_l1 / len(trainLoader), writer_iter)

        if e % args.saving_epoch == 0:

            val_times = 10

            val_saving_folder = pjoin(predictions_path, '{:04d}'.format(e))
            os.makedirs(val_saving_folder)

            val_Loss = 0

            with torch.no_grad():
                for val_step, data in tqdm(enumerate(valLoader)):

                    if val_step == val_times:
                        break

                    input, extras, gt = toolFuncs.Process_Input(data, type=config.dataType)
                    for key in extras:
                        extras[key] = extras[key].cuda()

                    pred = model(input)

                    loss = criterion(pred, extras, gt)

                    data = toolFuncs.Postprocess(data, pred, type=config.dataType)

                    val_Loss += loss.item()

                    val_saving_subdir = pjoin(val_saving_folder, '{:02d}'.format(val_step))
                    os.makedirs(val_saving_subdir)

                    for key in data:
                        img = utils.tensorToNumpy(data[key][0])
                        utils.save_exr(img, pjoin(val_saving_subdir, "{}.exr".format(key)))

                val_Loss /= val_times
                writer.add_scalar("val/Loss", val_Loss, writer_iter)
        
            
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'writer_iter' : writer_iter
            }, pjoin(val_saving_folder, 'checkpoint.pth'))

            
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'writer_iter' : writer_iter
            }, pjoin(model_saving_path, 'checkpoint.pth'))
        




if __name__ == '__main__':
    train()
    print('done')