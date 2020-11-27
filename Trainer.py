import torch
import metrics as m
from Config import Config
from Dataset import *

import torch.optim as optim
import torch.nn as nn


def train_model(
        dataset: SymbolsDataset,
        optimizer,
        loss_func,
        model
):
    num_of_train_samples = dataset.get_num_of_train_samples()
    num_of_test_samples = dataset.get_num_of_test_samples()

    num_of_train_samples = num_of_train_samples // Config.batch_size

    for epoch in range(Config.num_of_epochs):
        print()
        print('Epoch #', epoch + 1, '/', Config.num_of_epochs)

        # loss data
        train_loss = 0
        test_loss = 0

        for i in tqdm(range(num_of_train_samples)):
            batch_indices = [i for i in range(i * Config.batch_size, (i + 1) * Config.batch_size)]
            sample, labels = dataset.get_train_samples(batch_indices)

            out = model(sample)

            loss = loss_func(labels, out)
            loss.backward()
            optimizer.step()

            train_loss += loss.tolist()

        with torch.no_grad():
            for i in tqdm(range(num_of_test_samples)):
                sample, labels = dataset.get_test_samples([i])
                out = model(sample)
                loss = loss_func(labels, out)

                test_loss += loss.tolist()

        avg_train_loss = train_loss / num_of_train_samples
        avg_test_loss = test_loss / num_of_test_samples

        print()
        print('------------------------------------------')
        print('Average train loss:  ', "{:.04f}".format(avg_train_loss))
        print('Average test loss:   ', "{:.04f}".format(avg_test_loss))
