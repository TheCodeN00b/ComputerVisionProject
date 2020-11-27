import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as f

import numpy as np
from numpy import asarray
from PIL import Image, ImageFilter

import os

import Config as Conf
from Config import Config as Conf_var
from tqdm import tqdm

from random import seed
from random import random
import random

import calendar
import time
ts = calendar.timegm(time.gmtime())

# seed random number generator
seed(ts)


class SymbolsDataset(Dataset):
    def __init__(
            self,
            frac=1.0
    ):
        self.train = True

        print('Reading dataset..')

        # For each sample the system extracts a random number in [0, 1]. If greater then this value it will be used
        # for the training, otherwise for the test
        threshold = 0.2

        data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ])

        # Data tensors
        self.train_symbol_images = torch.FloatTensor([])
        self.train_labels = torch.FloatTensor([])

        self.test_symbol_images = torch.FloatTensor([])
        self.test_labels = torch.FloatTensor([])

        # List of directories
        dirs = os.listdir(Conf_var.dataset_filepath)

        for directory in dirs:
            print('Reading symbol ' + directory)
            symbols = os.listdir(Conf_var.dataset_filepath + '/' + directory)

            num_of_symbols_in_dir = len(symbols)
            num_of_symbols_to_take = int(num_of_symbols_in_dir * frac)

            indices = random.sample(range(0, num_of_symbols_in_dir - 1), num_of_symbols_to_take)

            for i in tqdm(range(num_of_symbols_to_take)):
                symbol = symbols[indices[i]]

                if symbol[-4:] == '.jpg':
                    # image symbol as a tensor
                    img_symbol = data_transform(Image.open(Conf_var.dataset_filepath + '/' + directory + '/' + symbol)).view(1, 1, 48, 48)

                    # it creates the vector for one-hot encoding
                    label_vector = torch.zeros((1, Conf_var.classes))
                    label_vector[:, Conf.symbol_to_idx[directory]] = 1

                    # Random number is extracted
                    coin = random.random()
                    if coin > threshold:
                        self.train_symbol_images = torch.cat((self.train_symbol_images, img_symbol), dim=0)
                        self.train_labels = torch.cat((self.train_labels, label_vector), dim=0)
                    else:
                        self.test_symbol_images = torch.cat((self.test_symbol_images, img_symbol), dim=0)
                        self.test_labels = torch.cat((self.test_labels, label_vector), dim=0)

    def get_num_of_train_samples(self):
        return len(self.train_symbol_images)

    def get_num_of_test_samples(self):
        return len(self.test_symbol_images)

    def get_train_samples(self, item):
        return self.train_symbol_images[item], self.train_labels[item]

    def get_test_samples(self, item):
        return self.test_symbol_images[item], self.test_labels[item]

    def info(self):
        print('Train dataset length:', len(self.train_symbol_images))
        print('Test dataset length:', len(self.test_symbol_images))
