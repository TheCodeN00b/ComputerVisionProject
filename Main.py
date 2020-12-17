from Dataset import *
from Config import Config as Conf
from Model import *

import torch
import torchvision
from torchviz import make_dot
import torch.optim as optim
import torchvision.transforms as transforms

import metrics as m
import Utils as u

import Trainer as t

from tqdm import tqdm


def run_confusion_matrix_test(test_dataset):
    model = Conv1DSymbolDetection()

    if Conf.use_cuda:
        model.to('cuda')

    checkpoint = torch.load('model/symbol_detector.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    predicted_labels = torch.zeros((len(test_dataset), Conf.classes)).to('cuda' if Conf.use_cuda else 'cpu')
    target_labels = torch.zeros(len(test_dataset)).to('cuda' if Conf.use_cuda else 'cpu')

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            sample, target = test_dataset[i: i + 1]

            out = model(sample)
            predicted_labels[i] = out
            target_labels[i] = target

    utils.print_confusion_matrix(predicted_labels, target_labels)


if __name__ == '__main__':
    # print('[Main]')
    #
    # # train_loader = torch.utils.data.DataLoader(
    # #     torchvision.datasets.MNIST('/files/', train=True, download=True,
    # #                                transform=torchvision.transforms.Compose([
    # #                                    torchvision.transforms.Grayscale(),
    # #                                    torchvision.transforms.ToTensor(),
    # #                                    torchvision.transforms.Normalize(
    # #                                        (0.1307,), (0.3081,))
    # #                                ])),
    # #     batch_size=Conf.train_dataset_size, shuffle=True)
    # #
    # # test_loader = torch.utils.data.DataLoader(
    # #     torchvision.datasets.MNIST('/files/', train=False, download=True,
    # #                                transform=torchvision.transforms.Compose([
    # #                                    torchvision.transforms.Grayscale(),
    # #                                    torchvision.transforms.ToTensor(),
    # #                                    torchvision.transforms.Normalize(
    # #                                        (0.1307,), (0.3081,))
    # #                                ])),
    # #     batch_size=int(Conf.test_dataset_size), shuffle=True)
    # #
    # # train_dataset = MNISTDataset(train_loader)
    # # test_dataset = MNISTDataset(test_loader)
    #
    train_dataset = SymbolsDataset(Conf.train_dataset_filepath, size=Conf.train_dataset_size)
    train_dataset.balance_dataset()
    #
    test_dataset = SymbolsDataset(Conf.test_dataset_filepath, size=Conf.test_dataset_size)
    test_dataset.balance_dataset()
    # #
    # # trans = transforms.ToPILImage()
    # # img, target = train_dataset[0, 0]
    # # trans(img.to('cpu')).show()
    # # # trans(target.to('cpu')).show()
    # # #
    model = Conv1DSymbolDetection()
    if Conf.use_cuda:
        model.to('cuda')
    #
    t.train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss_func=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(lr=1e-4, params=model.parameters(), momentum=0.8)
    )
    #
    # # trans = transforms.Compose([transforms.ToPILImage(), transforms.Normalize((-0.1307 * 0.229,), (-0.3081 * 0.299,))])
    # # img, target = train_dataset[0, 0]
    # # out = model(img)
    # #
    # # trans(img.to('cpu')).show()
    # # trans(target.to('cpu')).show()
    # # trans(out.to('cpu')).show()
    #
    # # sample, targets = train_dataset[0: 2]
    # # out = model(sample)
    # # vis_graph = make_dot(out, params={**{'inputs': sample}, **dict(model.named_parameters())})
    # # vis_graph.view()

    run_confusion_matrix_test(test_dataset)
