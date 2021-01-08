"""
The class manages all the hyper-parameters.
"""

from dataclasses import dataclass


@dataclass()
class Config:

    # ------------------------ constants --------------------------------
    # nope.
    # ------------------------- shared parameters --------------------------------------------

    img_h = 215  # height of the input images
    img_w = 450  # width of the input images
    use_gpu = True
    saveTrainingOutput = False

    # ------------------------ Denoising autoencoder parameters --------------------------------

    a_datasetPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/dataset/'
    a_modelsPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/models/'
    a_plotPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/plot/'
    DNet_phase = 'train'
    num_of_epochs = 100  # number of epochs
    batch_size = 4
    dNet_learning_rate = 1e-3
    dNet_validation_split = 0.20
    dNet_skip_training = True
    dNet_resumeTraining = False