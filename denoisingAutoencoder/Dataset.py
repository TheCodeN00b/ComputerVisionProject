"""
The class manages the dataset and all of its functionalities.
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy import asarray
from PIL import Image
from denoisingAutoencoder.Config import Config as Conf
from denoisingAutoencoder import Utils


class DenoisingDataset(Dataset):
    """
    The class manages the images dataset. When it is created it reads from  a filepath a list of images which are read
    and stored as a collection of tuple (noised_image, clean_image)
    """

    def __init__(
            self
            , path: str
    ):

        self.path = path
        self.dataset_images = list()

        images = self.read_images()

        for i in range(len(images)):
            #print(images[i].shape)
            self.dataset_images.append(images[i])

    def read_images(self):

        images_directory = self.path
        images = os.listdir(images_directory + '/noisy')
        images_list = []

        for img in images:

            # we read clean and noisy image from the dataset directory
            noisy_image = cv2.imread(images_directory + '/noisy/' + img, )
            clean_image = cv2.imread(images_directory + '/clean/' + img, )

            # We convert to one channel images
            noisy_gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
            clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)

            # We resize each image
            noisy_gray = cv2.resize(noisy_gray, (Conf.img_w, Conf.img_h))
            noisy_gray = torch.from_numpy(noisy_gray).unsqueeze(0).float()

            clean_gray = cv2.resize(clean_gray, (Conf.img_w, Conf.img_h))
            clean_gray = torch.from_numpy(clean_gray).unsqueeze(0).float()

            tuple = (noisy_gray, clean_gray)

            images_list.append(tuple)

        #numpy.random.shuffle(images_np)
        #train, validation = images_np[:80, :], images_np[80:, :]

        return images_list

    def __getitem__(self, index):
        return self.dataset_images[index]

    def __len__(self):
        return len(self.dataset_images)

