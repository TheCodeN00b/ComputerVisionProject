from Dataset import *
from Config import Config as Conf
from Model import *

import torch
import torchvision
from torchviz import make_dot
import torch.optim as optim

import metrics as m
import Utils as u

import Trainer as t


if __name__ == '__main__':
    print('[Main]')

    # train_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=True, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=Conf.train_dataset_size, shuffle=True)
    #
    # test_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('/files/', train=False, download=True,
    #                                transform=torchvision.transforms.Compose([
    #                                    torchvision.transforms.ToTensor(),
    #                                    torchvision.transforms.Normalize(
    #                                        (0.1307,), (0.3081,))
    #                                ])),
    #     batch_size=int(Conf.test_dataset_size), shuffle=True)
    #
    # train_dataset = MNISTDataset(train_loader)
    # test_dataset = MNISTDataset(test_loader)

    train_dataset = SymbolsDataset(Conf.train_dataset_filepath)
    test_dataset = SymbolsDataset(Conf.test_dataset_filepath)

    model = Conv1DSymbolDetection()

    if Conf.use_cuda:
        model.to('cuda')

    print(model)

    t.train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss_func=nn.BCEWithLogitsLoss(),
        optimizer=optim.SGD(lr=1e-1, params=model.parameters(), weight_decay=5e-4)
    )

    # sample, targets = train_dataset[0: 2]
    # out = model(sample)
    # vis_graph = make_dot(out, params={**{'inputs': sample}, **dict(model.named_parameters())})
    # vis_graph.view()
