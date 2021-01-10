import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as f

import numpy as np
from numpy import asarray
from PIL import Image, ImageFilter

import os
import shutil

import model.Config as Conf
from model.Config import Config as Conf_var
from tqdm import tqdm

from random import seed
from random import random
import random

import calendar
import time

ts = calendar.timegm(time.gmtime())

# seed random number generator
seed(ts)


def cleanse_dataset():
    train_dataset_filepath = Conf.Config.train_dataset_filepath + '_bold'
    test_dataset_filepath = Conf.Config.test_dataset_filepath + '_bold'

    golden_imgs = 'C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/best_samples'

    train_symbols = os.listdir(train_dataset_filepath)
    test_symbols = os.listdir(test_dataset_filepath)

    symbols_rejected = {
        '(': 0,
        ')': 0,
        '+': 0,
        '-': 0,
        '0': 0,
        '1': 0,
        '2': 0,
        '3': 0,
        '4': 0,
        '5': 0,
        '6': 0,
        '7': 0,
        '8': 0,
        '9': 0,
        'log': 0,
        'sqrt': 0,
        'x': 0
    }

    data_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((Conf.Config.img_size, Conf.Config.img_size)),
        transforms.ToTensor(),
    ])

    for i in tqdm(range(len(train_symbols))):
        symbol = train_symbols[i]
        try:
            img = data_transform(Image.open(train_dataset_filepath + '/' + symbol)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)
            if 'sqrt' in symbol:
                symbol_name = 'sqrt'
            elif 'log' in symbol:
                symbol_name = 'log'
            else:
                symbol_name = symbol[0]

            # if symbol_name == 'log' or symbol_name == 'sqrt':
            # golden_img = data_transform(Image.open(golden_imgs + '/' + symbol_name + '.jpg')).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)
            # l1_diff = torch.nn.functional.l1_loss(img, golden_img).tolist()

            flat_img = img.view(32 * 32)
            white_pixels = flat_img[flat_img[:] == 1].size()[0]
            fraction = white_pixels / (32 * 32)

            if fraction > 0.9:
                    # (l1_diff > 0.24 and symbol_name in ['(', ')']) or (l1_diff > 0.35 and symbol_name in ['x']) or\
                    # (l1_diff > 0.3 and symbol_name in ['1', '2']) or (l1_diff > 0.4) or \:
                symbols_rejected[symbol_name] += 1
                os.remove(train_dataset_filepath + '/' + symbol)

        except Exception as e:
            x = 1
            print(e)

    for key in symbols_rejected.keys():
        print('Rejected', symbols_rejected[key], 'images for symbol', key)


def build_bold_dataset():
    # creating data directories
    # os.mkdir('C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/train_data_bold')
    # os.mkdir('C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/test_data_bold')

    train_symbols = os.listdir(Conf.Config.train_dataset_filepath)
    test_symbols = os.listdir(Conf.Config.test_dataset_filepath)

    data_transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Pad(padding=(14, 14), fill=255, padding_mode='constant'),
        transforms.ColorJitter(contrast=1000),
        transforms.Resize((Conf.Config.img_size, Conf.Config.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    symbols_to_skip = []

    # for i in tqdm(range(len(train_symbols))):
    #     symbol = train_symbols[i]
    #     try:
    #         img = data_transform(Image.open(Conf.Config.train_dataset_filepath + '/' + symbol)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)
    #         bold_img = img
    #
    #         pixels = []
    #         for i in range(28 * 28):
    #             if bold_img[0, 0, i // 28, i % 28] != 1:
    #                 pixels.append((i // 28, i % 28))
    #
    #         if len(pixels) / (28 * 28) > 0.8 or len(pixels) / (28 * 28) < 0.01:
    #             symbols_to_skip.append(symbol)
    #
    #         for pixel in pixels:
    #             i, j = pixel
    #             bold_img[0, 0, i, j] = 0
    #             bold_img[0, 0, min(27, i + 1), min(27, j + 1)] = 0
    #
    #         pil_img = transforms.ToPILImage(mode='L')(bold_img[0, 0].to('cpu'))
    #         pil_img.save('C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/train_data_bold/' + symbol)
    #     except:
    #         x = 1

    # symbols_to_skip = []
    #
    for i in tqdm(range(len(train_symbols))):
        symbol = train_symbols[i]
        if 'log' in symbol:
            try:
                img = data_transform(Image.open(Conf.Config.train_dataset_filepath + '/' + symbol)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)
                bold_img = img

                pixels = []
                for i in range(28):
                    for j in range(28):
                        if bold_img[0, 0, i, j] != 1:
                            pixels.append((i, j))

                for pixel in pixels:
                    i, j = pixel
                    bold_img[0, 0, i, j] = 0
                    bold_img[0, 0, min(27, i + 1), min(27, j + 1)] = 0

                pil_img = transforms.ToPILImage(mode='L')(bold_img[0, 0].to('cpu'))
                pil_img.save('C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/train_data_bold/' + symbol)
            except:
                x = 1

    # print(len(symbols_to_skip))
    # print(symbols_to_skip)

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
            filepath,
            size=-1
    ):
        print('Reading dataset at', filepath)

        data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((Conf.Config.img_size, Conf.Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.new_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.RandomRotation((-15, 15)),
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
            symbol_name = ''
            if 'log' in symbol_filepath:
                symbol_name = 'log'
            elif 'sqrt' in symbol_filepath:
                symbol_name = 'sqrt'
            else:
                symbol_name = symbol_filepath[0]

            try:
                img_symbol = data_transform(Image.open(filepath + '/' + symbol_filepath)).view(1, 1, Conf.Config.img_size, Conf.Config.img_size)
                label_code = Conf.symbol_to_idx[symbol_name]

                # # it creates the vector for one-hot encoding
                # label_vector = torch.zeros((1, Conf_var.classes))
                # label_vector[:, Conf.symbol_to_idx[symbol_name]] = 1

                self.data[i] = img_symbol
                self.labels[i] = label_code
            except:
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
            num_of_elements = (self.labels == Conf.symbol_to_idx[label]).nonzero().size()[0]
            if num_of_elements > max_number_of_elements:
                max_number_of_elements = num_of_elements
                most_frequent_class = label

        # finds classes to upsamples
        for label in Conf.class_names:
            if label != most_frequent_class:
                num_of_elements = (self.labels == Conf.symbol_to_idx[label]).nonzero().size()[0]
                fraction = num_of_elements / max_number_of_elements

                if fraction < 1 / 2:
                    classes_to_upsample.append(label)

        for label in classes_to_upsample:
            label_elements = (self.labels == Conf.symbol_to_idx[label]).nonzero()
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
        print('# of images of 0:        ', (self.labels == Conf.symbol_to_idx['0']).nonzero().size()[0])
        print('# of images of 1:        ', (self.labels == Conf.symbol_to_idx['1']).nonzero().size()[0])
        print('# of images of 2:        ', (self.labels == Conf.symbol_to_idx['2']).nonzero().size()[0])
        print('# of images of 3:        ', (self.labels == Conf.symbol_to_idx['3']).nonzero().size()[0])
        print('# of images of 4:        ', (self.labels == Conf.symbol_to_idx['4']).nonzero().size()[0])
        print('# of images of 5:        ', (self.labels == Conf.symbol_to_idx['5']).nonzero().size()[0])
        print('# of images of 6:        ', (self.labels == Conf.symbol_to_idx['6']).nonzero().size()[0])
        print('# of images of 7:        ', (self.labels == Conf.symbol_to_idx['7']).nonzero().size()[0])
        print('# of images of 8:        ', (self.labels == Conf.symbol_to_idx['8']).nonzero().size()[0])
        print('# of images of 9:        ', (self.labels == Conf.symbol_to_idx['9']).nonzero().size()[0])
        # print('# of images of log:      ', (self.labels == Conf.symbol_to_idx['log']).nonzero().size()[0])
        print('# of images of sqrt:     ', (self.labels == Conf.symbol_to_idx['sqrt']).nonzero().size()[0])
        print('# of images of x:        ', (self.labels == Conf.symbol_to_idx['x']).nonzero().size()[0])
        print('# of images of (:        ', (self.labels == Conf.symbol_to_idx['(']).nonzero().size()[0])
        print('# of images of ):        ', (self.labels == Conf.symbol_to_idx[')']).nonzero().size()[0])
        print('# of images of +:        ', (self.labels == Conf.symbol_to_idx['+']).nonzero().size()[0])
        print('# of images of -:        ', (self.labels == Conf.symbol_to_idx['-']).nonzero().size()[0])
        print('------------------------------------')
        print('Total number of elements:', len(self.data))