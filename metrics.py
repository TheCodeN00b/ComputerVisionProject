import torch
from Config import Config as Conf
import Utils


def compute_accuracy(target_labels: torch.FloatTensor, predicted_labels: torch.FloatTensor):
    predicted_labels = Utils.convert_model_output(predicted_labels)
    target_labels_indices = target_labels[:] == 1

    total = target_labels.size()[0]
    true_positives = predicted_labels[target_labels_indices]

    correct = true_positives[true_positives == 1].size()[0]
    accuracy = (correct / total) * 100

    print()
    print('Total:       ', total)
    print('Correct:     ', correct)
    print('---------------------')
    print('Accuracy:    ', "{:0.4f}".format(accuracy) + '%')
