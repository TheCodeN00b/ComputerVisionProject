import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as f

import numpy as np
from numpy import asarray
from PIL import Image, ImageFilter

import os
import shutil

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


def build_dataset():
    # creating data directories
    os.mkdir('train_data')
    os.mkdir('test_data')

    # List of directories
    dirs = os.listdir(Conf_var.train_dataset_filepath)
    for directory in dirs:
        print('Reading symbol ' + directory)
        symbols = os.listdir(Conf_var.train_dataset_filepath + '/' + directory)

        num_of_symbols = len(symbols)
        random_indices = list(range(0, num_of_symbols))
        random.shuffle(random_indices)

        train_symbol_indices = random_indices[0: int(0.8 * num_of_symbols)]
        test_symbol_indices = random_indices[int(0.8 * num_of_symbols): num_of_symbols]

        # reading train data
        for i in tqdm(range(len(train_symbol_indices))):
            # copying the symbol to the new directory
            symbol = symbols[train_symbol_indices[i]]
            shutil.copy(Conf_var.train_dataset_filepath + '/' + directory + '/' + symbol, 'train_data/' + directory + '_' + str(i) + '.jpg')

        # reading train data
        for i in tqdm(range(len(test_symbol_indices))):
            # copying the symbol to the new directory
            symbol = symbols[test_symbol_indices[i]]
            shutil.copy(Conf_var.train_dataset_filepath + '/' + directory + '/' + symbol, 'test_data/' + directory + '_' + str(i) + '.jpg')


class MNISTDataset(Dataset):
    def __init__(self, loader):
        examples = enumerate(loader)
        batch_idx, (data, targets) = next(examples)

        self.data = data
        self.targets = torch.zeros((targets.size()[0], Conf.Config.classes))

        for i in tqdm(range(targets.size()[0])):
            target = targets[i].tolist()
            idx = Conf.symbol_to_idx[str(target)]
            self.targets[i, idx] = 1

    def __getitem__(self, item):
        return self.data[item].to('cuda' if Conf.Config.use_cuda else 'cpu'), self.targets[item].to('cuda' if Conf.Config.use_cuda else 'cpu')

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        indices = list(range(0, len(self.data)))
        random.shuffle(indices)

        for i in range(len(indices)):
            self.data[i] = self.data[indices[i]]
            self.targets[i] = self.targets[indices[i]]


class SymbolsDataset(Dataset):
    def __init__(
            self,
            filepath
    ):
        print('Reading dataset at', filepath)

        data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((Conf.Config.img_size, Conf.Config.img_size)),
            transforms.ToTensor(),
        ])

        # List of directories
        symbols = os.listdir(filepath)

        # Data tensors
        self.data = torch.randn((len(symbols), 1, Conf.Config.img_size, Conf.Config.img_size))
        self.labels = torch.randn((len(symbols), Conf.Config.classes))

        for i in tqdm(range(len(symbols))):
            symbol_filepath = symbols[i]
            symbol_name = symbol_filepath[0]

            try:
                img_symbol = data_transform(Image.open(filepath + '/' + symbol_filepath)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)

                # it creates the vector for one-hot encoding
                label_vector = torch.zeros((1, Conf_var.classes))
                label_vector[:, Conf.symbol_to_idx[symbol_name]] = 1

                self.data[i] = img_symbol
                self.labels[i] = label_vector
            except:
                x = 1

    def __getitem__(self, item):
        return self.data[item].to('cuda' if Conf.Config.use_cuda else 'cpu'), self.labels[item].to('cuda' if Conf.Config.use_cuda else 'cpu')

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        indices = list(range(0, len(self.data)))
        random.shuffle(indices)

        for i in range(len(indices)):
            self.data[i] = self.data[indices[i]]
            self.labels[i] = self.labels[indices[i]]
