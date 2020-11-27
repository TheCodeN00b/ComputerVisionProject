from Dataset import SymbolsDataset
from Config import Config as Conf
from Model import *

import torch
from torchviz import make_dot
import torch.optim as optim

import metrics as m
import Utils as u

import Trainer as t


if __name__ == '__main__':
    print('[Main]')

    dataset = SymbolsDataset(frac=0.02)

    sample, labels = dataset.get_train_samples([i for i in range(3)])
    dataset.info()
    model = SymbolDetector()

    t.train_model(
        dataset=dataset,
        model=model,
        loss_func=nn.MSELoss(),
        optimizer=optim.Adam(lr=1e-3, params=model.parameters())
    )

    out = model(sample)
    vis_graph = make_dot(out, params={**{'inputs': sample}, **dict(model.named_parameters())})
    vis_graph.view()
