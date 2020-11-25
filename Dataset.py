import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as f

import numpy as np
from numpy import asarray
from PIL import Image, ImageFilter

import os
import random

import Config as Conf
from Config import Config as Conf_var
from tqdm import tqdm

from random import seed
from random import random

import calendar
import time
ts = calendar.timegm(time.gmtime())

# seed random number generator
seed(ts)


class SymbolsDataset(Dataset):
    def __init__(
            self,
            train=True
    ):
        print('Reading dataset..')

        prob_treshold = 0
        if train:
            prob_treshold = 0.2
        else:
            prob_treshold = 0.8

        data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        self.symbol_images = torch.FloatTensor([])
        self.labels = torch.FloatTensor([])

        dirs = os.listdir(Conf_var.dataset_filepath)
        for dir in dirs:
            print('Reading symbol ' + dir)
            symbols = os.listdir(Conf_var.dataset_filepath + '/' + dir)
            for i in tqdm(range(len(symbols))):
                symbol = symbols[i]
                # image symbol as a tensor
                img_symbol = data_transform(Image.open(Conf_var.dataset_filepath + '/' + dir + '/' + symbol))

                coin = random()
                if coin > prob_treshold:
                    # image is put into the dataset
                    label_vector = torch.zeros(Conf_var.classes)
                    label_vector[Conf.symbol_to_idx[dir]] = 1

                    self.symbol_images = torch.cat((self.symbol_images, img_symbol), dim=0)
                    self.labels = torch.cat((self.labels, label_vector))

    def info(self):
        print('Dataset length:', len(self.symbol_images))
