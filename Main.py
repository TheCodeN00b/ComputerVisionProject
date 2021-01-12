from Dataset import *
from model.Config import Config as Conf
from model.Model import *
from denoisingAutoencoder.DenoisingNetwork import *

import torch
import torchvision
from torchviz import make_dot
import torch.optim as optim
import torchvision.transforms as transforms

import metrics as m
import Utils as u

import Trainer as t

from images import JpgImageIO as JpgIO
from images import ImageExplorer
from sort import FunctionExtractions as F
from PIL import Image
import cv2

from tqdm import tqdm


def run_confusion_matrix_test(test_dataset, model):
    checkpoint = torch.load('model_checkpoint/' + 'le_net_5.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predicted_labels = torch.zeros((len(test_dataset), Conf.classes)).to('cuda' if Conf.use_cuda else 'cpu')
    target_labels = torch.zeros(len(test_dataset)).to('cuda' if Conf.use_cuda else 'cpu')

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            sample, target = test_dataset[i: i + 1]

            out, _ = model(sample)
            predicted_labels[i] = out
            target_labels[i] = target

    u.print_confusion_matrix(predicted_labels, target_labels)


if __name__ == '__main__':
    print('[Main]')

    # Loading denoiser
    denoiser = DenoisingNetwork()
    denoiser.to('cuda' if Conf.use_cuda else 'cpu')
    denoiser_checkpoint = torch.load('denoisingAutoencoder/models/denoisingNet_checkpoint.pt')
    denoiser.load_state_dict(denoiser_checkpoint['model_state_dict'])

    # opening equation image (change resolution if too high)
    equation_img = Image.open('denoisingAutoencoder/dataset/test_images/IMG_20210111_154824.jpg')
    equation_img = transforms.ToTensor()(transforms.Resize((500, 500))(transforms.Grayscale()(equation_img))).view(1, 1, 500, 500).to('cuda' if Conf.use_cuda else 'cpu')

    # denoising image
    with torch.no_grad():
        cleansed_img = denoiser(equation_img)

    # increasing contrast
    height = cleansed_img.size()[2]
    width = cleansed_img.size()[3]
    contrast_cleansed_img = torch.zeros((1, 1, height, width))
    cleansed_img = cleansed_img.to('cpu')
    for i in tqdm(range(height)):
        for j in range(width):
            if cleansed_img[0, 0, i, j] < 0.9:
                contrast_cleansed_img[0, 0, i, j] = 0
            else:
                contrast_cleansed_img[0, 0, i, j] = 1

    # building equation
    equation_img = JpgIO.open_jpg_image('test/equation_cleansed.jpg', bw=True)
    symbols, tree = F.create_function(equation_img)
    print()
    u.print_node(tree, 0)
