"""
The class manages the dataset and all of its functionalities.
"""
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from numpy import asarray
from PIL import Image
from denoisingAutoencoder.Config import Config as Conf


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
        self.grey_images_dataset = list()


    def add_image_to_dataset(self):
        # augmentItem call
        return 0

    def augmentItem(self, originalItem, index):
        return 0

    def __getitem__(self, index):
        return self.grey_images_dataset[index]

    def __setitem__(self, key, value):
        self.grey_images_dataset[key] = value

    def __len__(self):
        return len(self.grey_images_dataset)

