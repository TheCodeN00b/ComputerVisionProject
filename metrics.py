import torch
from Config import Config as Conf
import Utils as utils
import sys


def compute_one_hot_accuracy(target_labels: torch.FloatTensor, predicted_labels: torch.FloatTensor):
    # predicted_labels = utils.convert_model_output(predicted_labels)

    predicted_classes = torch.argmax(torch.nn.functional.log_softmax(predicted_labels, dim=1), dim=1)
    target_classes = target_labels

    correct = target_classes[target_classes == predicted_classes].size()[0]

    return correct


def compute_accuracy(target_labels: torch.FloatTensor, predicted_labels: torch.FloatTensor):
    total = target_labels.size()[0]
    predicted = torch.argmax(predicted_labels, dim=1)
    correct = target_labels[target_labels == predicted].size()[0]

    return correct
