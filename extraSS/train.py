
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
parser.add_argument('--log_dir', type=str, default='E:\\Data\\Logs', help='saving dir')
parser.add_argument('--info', type=str, default='', help='short description')

parser.add_argument('--num_works', type=int, default=4, help='data loading threads')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--bz', type=int, default=2, help='batch size')

parser.add_argument('--vis_step', type=int, default=100, help="add to tensorboard frequency")

parser.add_argument('--saving_epoch', type=int, default=40, help='saving steps')
parser.add_argument('--total_epoch', type=int, default=1000, help='total steps')

parser.add_argument('--input_channels', type=int, default=13, help="input channel number")
parser.add_argument('--output_channels', type=int, default=3, help="output channel number")


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
    model = ExtraNet.ExtraNet_noHistory(args.input_channels, args.output_channels)
    model.cuda()
    ###########################################################################

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


    # DataLoader
    trainList = dataloader.extraSS_Dataset(args.data_dir, args.data_name, config.Dataloader_Keys)
    trainLoader = DataLoader(trainList,
                                batch_size=args.bz,
                                shuffle=True,
                                num_workers=args.num_works,
                                drop_last=True,
                                collate_fn=dataloader.extraSS_Dataset_syndata)

    valList = dataloader.extraSS_Dataset(args.data_dir, args.data_name, config.Dataloader_Keys)
    valLoader = DataLoader(valList,
                                batch_size=1,
                                shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                collate_fn=dataloader.extraSS_Dataset_syndata)


    # Loss Functions
    criterion = Losses.mLoss()

    for e in tqdm(range(args.total_epoch)):

        Loss = 0

        for data in tqdm(trainLoader):

            writer_iter += 1

            input, extras, gt = toolFuncs.Process_Input(data)
            input = input.cuda()
            gt = gt.cuda()
            for key in extras:
                extras[key] = extras[key].cuda()

            pred = model(input)

            loss = criterion(pred, extras["mask"].cuda(), gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer_iter % args.vis_step == 0:
                writer.add_scalar('train/step_loss', loss.item(), writer_iter)


            Loss += loss.item()


        writer.add_scalar('train/Loss', Loss / len(trainLoader), writer_iter)

        if e % args.saving_epoch == 0:

            val_times = 10

            val_saving_folder = pjoin(predictions_path, '{:04d}'.format(e))
            os.makedirs(val_saving_folder)

            val_Loss = 0

            with torch.no_grad():
                for val_step, data in tqdm(enumerate(valLoader)):

                    if val_step == val_times:
                        break

                    input, extras, gt = toolFuncs.Process_Input(data)
                    input = input.cuda()
                    gt = gt.cuda()
                    for key in extras:
                        extras[key] = extras[key].cuda()

                    pred = model(input)

                    loss = criterion(pred, extras["mask"].cuda(), gt)

                    val_Loss += loss.item()

                    val_saving_subdir = pjoin(val_saving_folder, '{:02d}'.format(val_step))
                    os.makedirs(val_saving_subdir)

                    pred_color_img = utils.tensorToNumpy(pred[0])
                    utils.save_exr(pred_color_img, pjoin(val_saving_subdir, "Pred.exr"))
                    for key in data:
                        img = utils.tensorToNumpy(data[key][0])
                        utils.save_exr(img, pjoin(val_saving_subdir, "{}.exr".format(key)))

                val_Loss /= val_times
                writer.add_scalar("val/Loss", val_Loss, writer_iter)
        

        

            



    




if __name__ == '__main__':
    train()
    print('done')