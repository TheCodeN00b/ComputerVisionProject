
import torch.nn as nn
import torchvision.models as models

class DenoisingNetwork(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(DenoisingNetwork, self).__init__()

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3)),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.Dropout2d(0.4),

            nn.MaxPool2d((2, 2)),

        )


        # DECODER
        self.decoder = nn.Sequential(

            nn.Conv2d(128, 64, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Upsample(scale_factor=(2, 2), mode="nearest"),

            nn.Conv2d(64, 1, kernel_size=(2, 3), stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),

        )



    def forward(self, input_image):

        latent_space = self.encoder(input_image)

        generated_output = self.decoder(latent_space)

        #print("------net output size>>>>>>>  " + str(generated_output.size()))

        return generated_output