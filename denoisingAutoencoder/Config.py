"""
The class manages all the hyper-parameters.
"""

from dataclasses import dataclass


@dataclass()
class Config:

    # ------------------------ constants --------------------------------
    # nope.
    # ------------------------- shared parameters --------------------------------------------

    img_h = 144  # height of the input images
    img_w = 144  # width of the input images
    use_gpu = True
    saveTrainingOutput = True

    # ------------------------ Denoising autoencoder parameters --------------------------------

    a_datasetPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/dataset/'
    a_modelsPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/models/'
    a_plotPath = 'C:/Users/theje/PycharmProjects/ComputerVisionProject/denoisingAutoencoder/plot/'
    DNet_phase = 'train'
    num_of_epochs = 5  # number of epochs
    batch_size = 32
    dNet_learning_rate = 3e-4
    dNet_validation_split = 0.20
    dNet_skip_training = False
    dNet_resumeTraining = False