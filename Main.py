from Dataset import *
from model.Config import Config as Conf
from model.Model import *

import torch
import torchvision
from torchviz import make_dot
import torch.optim as optim
import torchvision.transforms as transforms

import metrics as m
import Utils as u

import Trainer as t

from tqdm import tqdm


def run_confusion_matrix_test(test_dataset, model):
    checkpoint = torch.load(Conf.symbol_detector_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    predicted_labels = torch.zeros((len(test_dataset), Conf.classes)).to('cuda' if Conf.use_cuda else 'cpu')
    target_labels = torch.zeros(len(test_dataset)).to('cuda' if Conf.use_cuda else 'cpu')

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            sample, target = test_dataset[i: i + 1]

            out = model(sample)
            predicted_labels[i] = out
            target_labels[i] = target

    u.print_confusion_matrix(predicted_labels, target_labels)


if __name__ == '__main__':
    print('[Main]')

    train_dataset = SymbolsDataset(Conf.train_dataset_filepath, size=Conf.train_dataset_size)
    train_dataset.balance_dataset()
    train_dataset.print_info()

    test_dataset = SymbolsDataset(Conf.test_dataset_filepath, size=Conf.test_dataset_size)
    test_dataset.balance_dataset()
    test_dataset.print_info()

    model = Conv2DSymbolDetector()
    if Conf.use_cuda:
        model.to('cuda')

    t.train_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model=model,
        loss_func=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(lr=1e-3, params=model.parameters(), momentum=0.9)
    )

    # # sample, targets = train_dataset[0: 2]
    # # out = model_checkpoint(sample)
    # # vis_graph = make_dot(out, params={**{'inputs': sample}, **dict(model_checkpoint.named_parameters())})
    # # vis_graph.view()

    run_confusion_matrix_test(test_dataset, model)
