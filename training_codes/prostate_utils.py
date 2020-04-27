import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import os
import sys
import torch.nn as nn
import cv2
# from skimage.color import hed2rgb, rgb2hed
import random
import glob
import argparse
from PIL import Image


class data_loader(Dataset):
    """
    Dataset to read image and label for training
    """
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)

        # print(img.size())
        # print(int(self.imgs[index][-5]))

        return img, int(self.imgs[index][-5])

    def __len__(self):
        return len(self.imgs)



def parallelize_model(model):
    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        cudnn.benchmark = True
    return model


def unparallelize_model(model):
    try:
        while 1:
            # to avoid nested dataparallel problem
            model = model.module
    except AttributeError:
        pass
    return model


def cvt_to_gpu(X):
    return Variable(X.cuda()) if torch.cuda.is_available() \
    else Variable(X)
