import torch
import metrics as m
from model.Config import Config
from Dataset import *

import torch.optim as optim
import torch.nn as nn

import Utils as utils
import math


def train_model(
        train_dataset,
        test_dataset,
        optimizer,
        loss_func,
        model
):
    num_of_train_samples = len(train_dataset)
    num_of_test_samples = len(test_dataset)

    num_of_batches = num_of_train_samples // Config.batch_size

    train_losses = []
    test_losses = []

    train_acc_vec = []
    test_acc_vec = []

    prev_train_loss = float("inf")
    prev_train_acc = 0
    prev_test_loss = float("inf")
    prev_test_acc = 0

    for epoch in range(Config.num_of_epochs):
        train_dataset.shuffle()

        print()
        print('Epoch #', epoch + 1, '/', Config.num_of_epochs)

        # loss data
        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        t = tqdm(range(num_of_batches))
        for i in t:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            sample, labels = train_dataset[i * Config.batch_size: (i + 1) * Config.batch_size]
            out, _ = model(sample)

            # loss = loss_func(out, labels.long())
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_acc += m.compute_one_hot_accuracy(predicted_labels=out, target_labels=labels)
            train_loss += loss.tolist()

        with torch.no_grad():
            for i in tqdm(range(num_of_test_samples)):
                sample, labels = test_dataset[i: i + 1]
                out, _ = model(sample)
                loss = loss_func(out, labels)

                test_loss += loss.tolist()
                test_acc += m.compute_one_hot_accuracy(predicted_labels=out, target_labels=labels)

        avg_train_loss = train_loss / num_of_batches
        train_acc = (train_acc / num_of_train_samples) * 100
        avg_test_loss = test_loss / num_of_test_samples
        test_acc = (test_acc / num_of_test_samples) * 100

        if math.isnan(avg_train_loss) or math.isnan(avg_test_loss):
            break

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        train_acc_vec.append(train_acc)
        test_acc_vec.append(test_acc)

        print()
        print('------------------------------------------')
        print(
            'Average train loss:      ', "{:.04f}".format(avg_train_loss).rjust(7),
            '  Prev loss: ', '{:0.4f}'.format(prev_train_loss).rjust(7),
            '  Delta:   ', '{:.04f}'.format(avg_train_loss - prev_train_loss).rjust(7))
        print(
            'Average train acc:       ', "{:.02f}%".format(train_acc).rjust(7),
            '  Prev acc:  ', '{:0.2f}'.format(prev_train_acc).rjust(7),
            '  Delta:   ', '{:.02f}'.format(train_acc - prev_train_acc).rjust(7))
        print(
            'Average test loss:       ', "{:.04f}".format(avg_test_loss).rjust(7),
            '  Prev loss: ', '{:0.4f}'.format(prev_test_loss).rjust(7),
            '  Delta:   ', '{:.04f}'.format(avg_test_loss - prev_test_loss).rjust(7))
        print(
            'Average test acc:        ', "{:.02f}%".format(test_acc).rjust(7),
            '  Prev acc:  ', '{:0.2f}'.format(prev_test_acc).rjust(7),
            '  Delta:   ', '{:.02f}'.format(test_acc - prev_test_acc).rjust(7))

        prev_train_acc = train_acc
        prev_test_acc = test_acc

        if avg_train_loss < prev_train_loss and avg_test_loss < prev_test_loss:
            prev_train_loss = avg_train_loss
            prev_test_loss = avg_test_loss

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'reconstruction_loss': test_loss
            },
                'model_checkpoint/' + Config.symbol_detector_filename)
            print('[VideoInterpolationTrainer] Saved model_checkpoint checkpoint')

    utils.print_plot(
        plot_1=train_losses,
        plot_2=test_losses,
        labels=['Train loss', 'Test loss'],
        x_label='Epochs',
        y_label='Loss',
        plot_filename='loss_plot_conv2d_balance',
        save=True
    )
    utils.print_plot(
        plot_1=train_acc_vec,
        plot_2=test_acc_vec,
        labels=['Train accuracy', 'Test accuracy'],
        x_label='Epochs',
        y_label='Accuracy',
        plot_filename='accuracy_plot_conv2d_balance',
        save=True
    )
