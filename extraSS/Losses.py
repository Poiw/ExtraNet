import torch
import numpy as np
import torch.nn as nn
from config import mLossHoleArgument,mLossHardArgument

l1Loss = nn.L1Loss()

class LossHoleArgument(nn.Module):
    def __init__(self):
        super(LossHoleArgument,self).__init__()

    def forward(self, input, mask, target):
        lossMask = 1 - mask[:,:3,:,:]
        lholeAugment = ( lossMask * torch.abs(input-target) ).sum() / ( lossMask.sum() + 30 )
        return lholeAugment

class LossHardArgument(nn.Module):
    def __init__(self,ratio=0.1):
        super(LossHardArgument,self).__init__()
        self.ratio = ratio

    def forward(self, input, target):
        n, c, h, w = input.shape
        val, ind = torch.topk(torch.abs(input - target).view(n,c,-1), k=int(h*w*self.ratio))
        return val.mean()

class mLoss(nn.Module):
    def __init__(self):
        super(mLoss,self).__init__()
        self.hole = LossHoleArgument()
        self.hard = LossHardArgument()

    def forward(self, input, mask, target):
        basicl1 = torch.abs(input-target).mean()
        if mLossHoleArgument:
            basicl1 += self.hole(input,mask,target) * mLossHoleArgument
        if mLossHardArgument:
            basicl1 += self.hard(input,target) * mLossHardArgument
        return basicl1

class Multireso_mLoss(nn.Module):
    def __init__(self, low_weight=0.5, high_weight=0.5):
        super(Multireso_mLoss, self).__init__()

        self.mLoss = mLoss()
        self.low_w = low_weight
        self.high_w = high_weight
        
    def forward(self, input, extras, target):

        low_pred = input["low"]
        high_pred = input["high"]

        low_gt = target["low"]
        high_gt = target["high"]

        low_loss = self.mLoss(low_pred, extras["mask"], low_gt)
        high_loss = self.mLoss(high_pred, extras["high_mask"], high_gt)
        # high_loss = l1Loss(high_pred, high_gt)

        return self.low_w * low_loss + self.high_w * high_loss
