"""
The class manages the dataset and all of its functionalities.
"""
import os
import cv2
import torch
import random
from keras_preprocessing.image import img_to_array
from torch.utils.data import Dataset
import numpy as np
from numpy import asarray
from PIL import Image
from denoisingAutoencoder.Config import Config as Conf
from denoisingAutoencoder import Utils
from  random import randint


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
            self.dataset_images.append(images[i])

    def read_images(self):

        images_directory = self.path
        images = os.listdir(images_directory)
        images_list = []

        # se stiamo facendo test leggiamo il dataset da quella cartella!
        if(Conf.dNet_skip_training or Conf.DNet_phase == 'test'):
            for img in images:
                # we read clean and noisy image from the dataset directory
                noisy_image = cv2.imread(images_directory + '/' + img)

                # We convert to one channel images
                noisy_gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

                # converting the image to array and normalizing the data
                noisy_gray = img_to_array(noisy_gray, dtype='float32') / 255.

                # We resize each image
                noisy_gray = cv2.resize(noisy_gray, (Conf.img_w, Conf.img_h))

                # We add the new tuple to the list
                tuple = (noisy_gray, noisy_gray)
                images_list.append(tuple)

        else:
            images = os.listdir(images_directory + '/noisy')

            for img in images:

                # we read clean and noisy image from the dataset directory
                noisy_image = cv2.imread(images_directory + '/noisy/' + img, )
                clean_image = cv2.imread(images_directory + '/clean/' + img, )

                # We convert to one channel images
                noisy_gray = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)
                clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)

                # converting the image to array and normalizing the data
                noisy_gray = img_to_array(noisy_gray, dtype='float32') / 255.
                clean_gray = img_to_array(clean_gray, dtype='float32') / 255.

                # We resize each image
                noisy_gray = cv2.resize(noisy_gray, (Conf.img_w, Conf.img_h))
                clean_gray = cv2.resize(clean_gray, (Conf.img_w, Conf.img_h))

                # We add the new tuple to the list
                tuple = (noisy_gray, clean_gray)
                images_list.append(tuple)

        return images_list

    def __getitem__(self, index):
        return self.dataset_images[index]

    def __len__(self):
        return len(self.dataset_images)

    # We produce x3 changed copies of the same sample to augment data
    def augmentItem(self, originalItem, index):
        # produce two augmented samples
        # We are augmentig data x3 with traslation, flip, zoom an shift

        for i in range(3):
            mode = randint(0, 10)
            item0 = self.transformImage(originalItem[0], mode)
            item1 = self.transformImage(originalItem[1], mode)
            new_tuple = (item0, item1)

            # I'm adding the modified tuple to the dataset
            self.dataset_images.append(new_tuple)

        # delete the original item and keep the three augmented samples
        self.dataset_images.pop(index)

        tuple0Index = len(self.dataset_images) - 3
        tuple1Index = len(self.dataset_images) - 2
        tuple2Index = len(self.dataset_images) - 1

        return tuple0Index, tuple1Index, tuple2Index

# SOME METHODS TO TRANSFORM EXTRACTED FRAMES TO GIVE MORE VARIABILITY ON TRAINING DATA----------------------------------

    def fill(self, img, h, w):
        img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
        return img

    def zoom(self, img, value):
        if value > 1 or value < 0:
            print('Value for zoom should be less than 1 and greater than 0')
            return img
        value = random.uniform(value, 1)
        h, w = img.shape[:2]
        h_taken = int(value*h)
        w_taken = int(value*w)
        h_start = random.randint(0, h-h_taken)
        w_start = random.randint(0, w-w_taken)
        img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
        img = self.fill(img, h, w)
        return img

    def brightness(self, img, low, high):
        value = random.uniform(low, high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def vertical_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = h*ratio
        if ratio > 0:
            img = img[:int(h-to_shift), :, :]
        if ratio < 0:
            img = img[int(-1*to_shift):, :, :]
        img = self.fill(img, h, w)
        return img

    def horizontal_shift(self, img, ratio=0.0):
        if ratio > 1 or ratio < 0:
            print('Value should be less than 1 and greater than 0')
            return img
        ratio = random.uniform(-ratio, ratio)
        h, w = img.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img = img[:, :int(w-to_shift), :]
        if ratio < 0:
            img = img[:, int(-1*to_shift):, :]
        img = self.fill(img, h, w)
        return img

    def transformImage(self, img, mode):
        # performing several image transformations

        rand = mode

        if rand > 6:
            # vertical flip
            img = cv2.flip(img, 0)
        if rand >= 5:
            # horizontal flip
            img = cv2.flip(img, 1)
        if rand < 5:
            # image zoom
            img = self.zoom(img, 0.5)
        if rand == 5 or rand == 8:
            # change brightness
            img = self.brightness(img, 0.5, 0.9)
        if rand <= 3:
            # vertical shift
            img = self.vertical_shift(img, 0.5)
        if rand == 2 or rand == 10:
            # horizontal shift
            img = self.horizontal_shift(img, 0.5)

        return img

#-----------------------------------------------------------------------------------------------------------------------