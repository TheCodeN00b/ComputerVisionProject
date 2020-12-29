
import os
import warnings

import torch
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from denoisingAutoencoder.Config import Config as Conf
from matplotlib import pyplot as plt


# Shows what device you are using to compute and the model_checkpoint number of parameters
from denoisingAutoencoder.DenoisingNetwork import DenoisingNetwork


def showDeviceUsage(model):
    print('\n-------------------------------------------------------------------------------------------------------')
    # Use GPU if available and we want to use it
    if Conf.use_gpu and torch.cuda.is_available():
        model.cuda()
        print('Model loaded on GPU.')
    else:
        print('Model loaded on CPU.')
    device = torch.device("cuda:0" if Conf.use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)
    print('-------------------------------------------------------------------------------------------------------\n')


# return the device we are using to compute
def getUsedDevice():
    #  use gpu if available
    return torch.device("cuda" if torch.cuda.is_available() and Conf.use_gpu else "cpu")


# remove the temporary output generated in the reconstruction directory
def cleanOutputDirectory():
    if not (Conf.dNet_resumeTraining or Conf.dNet_skip_training):
        # we delete the output files content with information about each epoch if we are not resuming the training
        file = open(Conf.a_modelsPath + "train_losses.txt", "r+")
        file.truncate(0)
        file.close()
        file = open(Conf.a_modelsPath + "validation_losses.txt", "r+")
        file.truncate(0)
        file.close()

    print('[Main] Deleting previous reconstructed frames')
    frames = os.listdir(Conf.a_datasetPath + 'reconstruction_results/')
    for frame in frames:
        os.remove(Conf.a_datasetPath + 'reconstruction_results/' + frame)

# Normalize and convert the dataset samples and return the model_checkpoint input
def normalizeSample(sample, caller):
    normalized_image = 0
    return normalized_image


# save the reconstructed_image versus original_image output for DenoisingNetwork
def saveOutput(reconstructed_image, expected_image, mode, count):

    reconstructed_image = reconstructed_image.cpu().detach().numpy()
    expected_image = expected_image.cpu().detach().numpy()

    # save the reconstructed versus original output plot for the sample
    plt.imsave(Conf.a_datasetPath + 'reconstruction_results/' + mode + str(count) + " _reconstructed_image.png", reconstructed_image)
    plt.imsave(Conf.a_datasetPath + 'reconstruction_results/' + mode + str(count) + "_expected_image.png", expected_image)

    torch.cuda.empty_cache()

# save the passed image
def print_image(image):

    #image = image.cpu().detach().numpy()

    # save the reconstructed versus original output plot for the sample
    plt.imsave(Conf.a_datasetPath + 'reconstruction_results/print_image.png', image)

    torch.cuda.empty_cache()

# We resume the trained model_checkpoint from the last checkpoint
def resumeFromCheckpoint():

    model = DenoisingNetwork().to(getUsedDevice())
    optimizer = torch.optim.Adam(model.parameters(), lr=Conf.cNet_learning_rate, weight_decay=1e-10)
    checkpoint = torch.load(Conf.a_modelsPath + 'denoisingNet_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    showDeviceUsage(model)

    return model, optimizer, last_epoch, loss


# We save each loss value we produce in training to plot the entire curve
# because we can perform many run sessions of a restored model_checkpoint
def saveLossValue(loss, mode):
    # we save a file with information on the executed epoch
    if mode == 'train':
        f = open(Conf.a_modelsPath + "train_losses.txt", "a")
        f.write(str(loss) + "\n")
        f.close()

    if mode == 'validate':
        f = open(Conf.a_modelsPath + "validation_losses.txt", "a")
        f.write(str(loss) + "\n")
        f.close()


# At each epoch we save the loss value in a file.txt, here we restore those information to plot
# This file is used specifically when we run the model_checkpoint on separate sessions
def readRecordedLosses(type):
    # define empty list
    losses = []

    # open file and read the content in a list
    with open(Conf.a_modelsPath + type + '_losses.txt', 'r') as filehandle:
        filecontents = filehandle.readlines()

        for line in filecontents:
            # remove linebreak which is the last character of the string
            current_loss = line[:-1]

            # add item to the list
            losses.append(float(current_loss))

    return losses


# We plot the loss of each executed epoch
def plotLosses(train_losses, validation_losses):

    plt.subplot

    # we save the resulting loss curve
    plt.plot(np.array(train_losses), 'r', marker='.', label="training loss")
    plt.plot(np.array(validation_losses), 'b', marker='.', label="validation loss")
    plt.legend(loc='upper right')

    plt.grid()
    plt.title('Training vs validation loss')
    plt.xlabel('epochs')
    plt.savefig(Conf.a_plotPath + 'DenoisingNet_TrainingLoss.png', dpi=150)
