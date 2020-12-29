import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from denoisingAutoencoder.DenoisingNetwork import DenoisingNetwork
from denoisingAutoencoder.Dataset import DenoisingDataset
from denoisingAutoencoder.Config import Config as Conf
from torch.autograd import Variable
from denoisingAutoencoder import Utils
from denoisingAutoencoder.Utils import normalizeSample, saveOutput


class ImageDenoisingTrainer:
    """
    Utility class for train the model_checkpoint
    """

    def __init__(
            self,
            model: DenoisingNetwork,
            optimizer,
            used_dataset: DenoisingDataset
    ):
        """

        :param model: the model_checkpoint to train for the image denoising task
        :param optimizer: the used optimizer
        :param train_dataset: the train dataset
        """

        self.model = model.cuda() if Conf.use_gpu else model.cpu()
        self.reconstruction_loss = nn.L1Loss()
        self.optimizer = optimizer
        self.dataset = used_dataset

    def trainAndValidate(self, start_epoch):

        # split the dataset into train and validation
        train_loader, validation_loader = self.splitDataset()

        print("train set size: " + str(len(train_loader.sampler)))
        print("validation set size: " + str(len(validation_loader.sampler)) + "\n")

        # the used loss function
        mse = nn.MSELoss().cuda() if Conf.use_gpu else nn.MSELoss()
        l1 = nn.L1Loss().cuda() if Conf.use_gpu else nn.L1Loss()
        l1smooth = nn.SmoothL1Loss().cuda() if Conf.use_gpu else nn.SmoothL1Loss()

        loss_function = mse

        best_loss = -1

        for epoch in range(Conf.num_of_epochs):
            epoch_loss = 0

            "### TRAINING ###"
            for data in enumerate(train_loader):
                # we read the image from data that has the form of (step, frame) tuple
                tuple = data[1]
                noised_image = tuple[0].to(Utils.getUsedDevice())
                expected_output = tuple[1].to(Utils.getUsedDevice())

                self.optimizer.zero_grad()  # the optimizer is reset

                # Train on the selected sample 'grey_denoised_image'
                clean_reconstructed = self.model(noised_image)

                # reconstruction error between the clean expected image (target) and the reconstructed one
                train_loss = loss_function(clean_reconstructed, expected_output)

                # maintain the best loss
                if epoch == -1:
                    best_loss = train_loss

                # We save the training output (one for each epoch), in the results directory
                if Conf.saveTrainingOutput:
                    saveOutput(clean_reconstructed, expected_output, 'train', start_epoch + epoch + 1)

                # add the mini-batch training loss to epoch loss
                epoch_loss += train_loss.item() * Conf.batch_size

                # compute the accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                self.optimizer.step()

                torch.cuda.empty_cache()

            epoch_batches = len(train_loader.sampler) / Conf.batch_size

            # compute the epoch training loss
            epoch_loss = epoch_loss / epoch_batches

            # We also save each loss value in a file, to manage a resumed model_checkpoint training
            Utils.saveLossValue(epoch_loss, 'train')

            # display the epoch training loss
            print("epoch: {}/{}, loss = {:.6f}".format(start_epoch + epoch + 1, start_epoch + Conf.num_of_epochs,
                                                       epoch_loss))

            torch.cuda.empty_cache()

            "### VALIDATE epoch ###"
            self.validate(validation_loader, loss_function, start_epoch, epoch + 1, best_loss)

        print('Training phase finished!\n')


    def validate(self, validation_loader, loss_function, start_epoch, epoch, best_loss):

        epoch_loss = 0
        saveModel = True

        for data in enumerate(validation_loader):

            # we read the image from data that has the form of (step, frame) tuple
            tuple = data[1]
            noised_image = tuple[0].to(Utils.getUsedDevice())
            expected_output = tuple[1].to(Utils.getUsedDevice())

            # Train on the selected sample 'grey_denoised_image'
            clean_reconstructed = self.model(noised_image)

            # reconstruction error between the clean expected image (target) and the reconstructed one
            validation_loss = loss_function(clean_reconstructed, expected_output)

            # We save the training output (one for each epoch), in the results directory
            if Conf.saveTrainingOutput:
                saveOutput(clean_reconstructed, expected_output, 'train', start_epoch + epoch + 1)

            # updating the best loss
            if validation_loss < best_loss:
                best_loss = validation_loss
                saveModel = True

            # add the mini-batch training loss to epoch loss
            epoch_loss += validation_loss.item() * Conf.batch_size

        epoch_batches = len(validation_loader.sampler) / Conf.batch_size

        # compute the epoch validation loss
        epoch_loss = epoch_loss / epoch_batches

        # if this model_checkpoint version provides the best loss in validation we save it
        if saveModel:
            # we save the last epoch checkpoint of the model_checkpoint
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': validation_loss,
            }, Conf.a_modelsPath + 'colorizationNet_checkpoint.pt')

        # We also save each loss value in a file, to manage a resumed model_checkpoint training
        Utils.saveLossValue(epoch_loss, 'validate')

        # display the epoch training loss
        print("validation loss = {:.6f}".format(epoch_loss) + "\n")

        "We define a random seed to manage samples shuffle and we split the " \
        "dataset into train and validation according with the declared validation size"

        def splitDataset(self):

            # Creating data indices for training and validation splits:
            random_seed = 42  # seed to randomly shuffle the dataset
            dataset_size = len(self.dataset)
            indices = list(range(dataset_size))

            split = int(np.floor(Conf.cNet_validation_split * dataset_size))

            # indices shuffle
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            augmentedTrainIndexes = list()
            # replace each original training item with two augmented variations
            for index in train_indices:
                originalItem = self.dataset.__getitem__(index)
                newItem0, newItem1 = self.dataset.augmentItem(originalItem, index)

                # add the two variation indexes to add them to the final train loader
                # using the train sampler
                augmentedTrainIndexes.append(newItem0)
                augmentedTrainIndexes.append(newItem1)

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(augmentedTrainIndexes)
            valid_sampler = SubsetRandomSampler(val_indices)

            # build the augmented train loader and the normal validation loader
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=Conf.batch_size, sampler=train_sampler,
                                                       num_workers=4, pin_memory=True)
            validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=Conf.batch_size,
                                                            sampler=valid_sampler, num_workers=4, pin_memory=True)

            # return train and validation iterator for the learning phase
            return train_loader, validation_loader


    "We define a random seed to manage samples shuffle and we split the " \
    "dataset into train and validation according with the declared validation size"
    def splitDataset(self):

        # Creating data indices for training and validation splits:
        random_seed = 42  # seed to randomly shuffle the dataset
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        split = int(np.floor(Conf.dNet_validation_split * dataset_size))

        # indices shuffle
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # build the augmented train loader and the normal validation loader
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=Conf.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=Conf.batch_size, sampler=valid_sampler, num_workers=4, pin_memory=True)

        # return train and validation iterator for the learning phase
        return train_loader, validation_loader