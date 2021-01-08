
import torch
import torch.optim as optim
from denoisingAutoencoder import Utils
from denoisingAutoencoder.DenoisingNetwork import DenoisingNetwork
from denoisingAutoencoder.Utils import saveOutput, normalizeSample, getUsedDevice, cleanOutputDirectory
from denoisingAutoencoder.ImageDenoisingTrainer import ImageDenoisingTrainer
from denoisingAutoencoder.Dataset import DenoisingDataset
from denoisingAutoencoder.Config import Config as Conf

if __name__ == '__main__':

    # We delete the last generated dataset if is required to generate a new one
    # At any run, we delete each reconstructed output in the results directory
    cleanOutputDirectory()

    """
    -----------------------------------------------
    TRAINING PHASE
    -----------------------------------------------
    """
    if not Conf.dNet_skip_training:

        # we read the train_set by loading it from the dataset directory
        train_set = DenoisingDataset(Conf.a_datasetPath + 'train_images')

        # INSTANTIATING THE AUTOENCODER
        model = DenoisingNetwork().to(getUsedDevice())
        Utils.showDeviceUsage(model)

        print("##### Training the denoising autoencoder #####\n")

        # Train the model_checkpoint and save it in the project directory

        # if we resume an old checkpoint, we will start from a specific epoch of a previous run
        # and we have to resume even the same optimizer
        start_epoch = 0

        Adam = optim.Adam(model.parameters(), lr=Conf.dNet_learning_rate, weight_decay=9e-5)
        SGD = optim.SGD(model.parameters(), lr=Conf.dNet_learning_rate, momentum=0.96)
        used_optimizer = Adam

        # if we want to continue a previous uncompleted training
        if Conf.dNet_resumeTraining:
            model, optimizer, start_epoch, loss = Utils.resumeFromCheckpoint()
            print('Finished loading checkpoint. Resuming from epoch ' + str(start_epoch) + ' with loss: ' + str(loss))
            print('The model_checkpoint will be trained on other ' + str(Conf.num_of_epochs) + ' epochs:')

        # Call the trainer with the DenoisingNetwork model_checkpoint and the specified optimizer and our dataset
        trainer = ImageDenoisingTrainer(model, used_optimizer, train_set)
        trainer.trainAndValidate(start_epoch)

    # Test the loaded model_checkpoint if you decide to skip the training phase
    else:
        model, optimizer, epoch, loss = Utils.resumeFromCheckpoint()
        print('Denoising model_checkpoint was loaded from a checkpoint (trained on ' + str(epoch) + 'epoch, with loss: ' + str(loss))

    # We read all the recorded losses and we plot the entire curve
    Utils.plotLosses(Utils.readRecordedLosses('train'), Utils.readRecordedLosses('validation'))

    """
    -----------------------------------------------
    TEST PHASE
    -----------------------------------------------
    """
    Conf.DNet_phase = 'test'

    # we read the test_set by loading a previous generated one or by extracting it from new videos in test_videos
    test_set = DenoisingDataset(Conf.a_datasetPath + 'test_images')

    print("\n##### Testing the denoising autoencoder #####\n")

    count = 0

    print("test set size: " + str(len(test_set)) + "\n")

    for step, sample in enumerate(test_set):

        grey_noised_image = torch.from_numpy(sample[0])
        grey_noised_image = grey_noised_image.float().to(Utils.getUsedDevice())

        # count the number of processed images
        count += 1

        # Compute the reconstruction
        clean_reconstructed = model(grey_noised_image.reshape(-1, 1, Conf.img_h, Conf.img_w))
        clean_reconstructed = clean_reconstructed.reshape(-1, Conf.img_h, Conf.img_w)

        # save a reconstructed vs original frame plot for each sample in the test set
        # saving the original target form in RGB as pytorch tensor
        saveOutput(clean_reconstructed, clean_reconstructed, 'test', count)

        print("reconstructed image -> " + str(count))

