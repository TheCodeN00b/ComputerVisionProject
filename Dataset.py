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
        self.targets = torch.zeros((len(data)))

        for i in tqdm(range(targets.size()[0])):
            target = targets[i].tolist()
            idx = Conf.symbol_to_idx[str(target)]
            self.targets[i] = idx

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
            filepath,
            size=-1
    ):
        print('Reading dataset at', filepath)

        data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Pad(padding=(14, 14), fill=255, padding_mode='constant'),
            transforms.ColorJitter(contrast=200),
            transforms.Resize((Conf.Config.img_size, Conf.Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.new_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((-15, 15)),
            transforms.ToTensor()
        ])

        # List of symbols
        symbols = os.listdir(filepath)
        indices = list(range(0, len(symbols)))
        if size != -1:
            indices = random.sample(range(0, len(indices)), size)

        # Data tensors
        self.data = torch.zeros((size if size != -1 else len(symbols), 1, Conf.Config.img_size, Conf.Config.img_size))
        self.labels = torch.zeros((size if size != -1 else len(symbols))).long()

        for i in tqdm(range(len(indices))):
            symbol_filepath = symbols[indices[i]]
            if 'log' in symbol_filepath:
                symbol_name = 'log'
            elif 'sqrt' in symbol_filepath:
                symbol_name = 'sqrt'
            else:
                symbol_name = symbol_filepath[0]

            try:
                img_symbol = data_transform(Image.open(filepath + '/' + symbol_filepath)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)

                self.data[i] = img_symbol
                self.labels[i] = Conf.symbol_to_idx[symbol_name]
            except Exception as e:
                x = 1

    def __getitem__(self, item):
        return self.data[item].to('cuda' if Conf.Config.use_cuda else 'cpu'), self.labels[item].to('cuda' if Conf.Config.use_cuda else 'cpu')

    def __len__(self):
        return len(self.data)

    def balance_dataset(self):
        max_number_of_elements = -float("inf")
        most_frequent_class = ''
        classes_to_upsample = []

        # finds the most popular class
        for label in Conf.class_names:
            num_of_elements = self.labels[self.labels == Conf.symbol_to_idx[label]].size()[0]
            if num_of_elements > max_number_of_elements:
                max_number_of_elements = num_of_elements
                most_frequent_class = label

        # finds classes to upsamples
        for label in Conf.class_names:
            if label != most_frequent_class:
                num_of_elements = self.labels[self.labels == Conf.symbol_to_idx[label]].size()[0]
                fraction = num_of_elements / max_number_of_elements

                if fraction < 1 / 2:
                    classes_to_upsample.append(label)

        for label in classes_to_upsample:
            label_elements = self.labels[self.labels == Conf.symbol_to_idx[label]]
            num_of_elements = label_elements.size()[0]
            elements_to_add = max_number_of_elements - num_of_elements

            if num_of_elements > 0:
                quantity = int(elements_to_add / num_of_elements)
                new_images = torch.zeros((quantity * num_of_elements, 1, Conf.Config.img_size, Conf.Config.img_size))
                new_labels = torch.ones((quantity * num_of_elements)).long() * Conf.symbol_to_idx[label]

                print('Upsampling class', label, 'with', quantity, 'elements for each element')

                # t = tqdm(range(num_of_elements))
                for i in range(num_of_elements):
                    generated_images = self.__augment_single_image(self.data[label_elements[i]], quantity)
                    new_images[i * quantity: (i + 1) * quantity] = generated_images
                # tqdm.close(t)

                self.data = torch.cat((self.data, new_images), dim=0)
                self.labels = torch.cat((self.labels, new_labels), dim=0)

    def __augment_single_image(self, img, quantity):
        size = Conf.Config.img_size
        return_tensor = torch.zeros((quantity, 1, size, size))

        for i in range(quantity):
            new_img = self.new_img_transform(img.view(size, size)).view(1, 1, size, size)
            return_tensor[i] = new_img

        return return_tensor

    def shuffle(self):
        indices = list(range(0, len(self.data)))
        random.shuffle(indices)

        for i in range(len(indices)):
            self.data[i] = self.data[indices[i]]
            self.labels[i] = self.labels[indices[i]]

    def print_info(self):
        print('Dataset info')
        print('------------------------------------')
        print('# of images of 0:        ', self.labels[self.labels == Conf.symbol_to_idx['0']].size()[0])
        print('# of images of 1:        ', self.labels[self.labels == Conf.symbol_to_idx['1']].size()[0])
        print('# of images of 2:        ', self.labels[self.labels == Conf.symbol_to_idx['2']].size()[0])
        print('# of images of 3:        ', self.labels[self.labels == Conf.symbol_to_idx['3']].size()[0])
        print('# of images of 4:        ', self.labels[self.labels == Conf.symbol_to_idx['4']].size()[0])
        print('# of images of 5:        ', self.labels[self.labels == Conf.symbol_to_idx['5']].size()[0])
        print('# of images of 6:        ', self.labels[self.labels == Conf.symbol_to_idx['6']].size()[0])
        print('# of images of 7:        ', self.labels[self.labels == Conf.symbol_to_idx['7']].size()[0])
        print('# of images of 8:        ', self.labels[self.labels == Conf.symbol_to_idx['8']].size()[0])
        print('# of images of 9:        ', self.labels[self.labels == Conf.symbol_to_idx['9']].size()[0])
        print('# of images of log:      ', self.labels[self.labels == Conf.symbol_to_idx['log']].size()[0])
        print('# of images of sqrt:     ', self.labels[self.labels == Conf.symbol_to_idx['sqrt']].size()[0])
        print('# of images of x:        ', self.labels[self.labels == Conf.symbol_to_idx['x']].size()[0])
        print('# of images of (:        ', self.labels[self.labels == Conf.symbol_to_idx['(']].size()[0])
        print('# of images of ):        ', self.labels[self.labels == Conf.symbol_to_idx[')']].size()[0])
        print('------------------------------------')
        print('Total number of elements:', len(self.data))
