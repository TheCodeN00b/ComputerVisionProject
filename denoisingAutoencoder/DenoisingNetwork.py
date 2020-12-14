
import torch.nn as nn
import torchvision.models as models

class DenoisingNetwork(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(DenoisingNetwork, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential()


        # DECODER
        # Convolutional layers and upsampling
        self.decoder = nn.Sequential()



    def forward(self, input_image):

        return 0