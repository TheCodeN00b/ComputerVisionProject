import torch


def compute_one_hot_accuracy(target_labels: torch.FloatTensor, predicted_labels: torch.FloatTensor):
    predicted_classes = torch.argmax(predicted_labels, dim=1)
    target_classes = target_labels

    correct = 0
    for i in range(len(predicted_classes)):
        pred = predicted_classes[i].tolist()
        target = target_classes[i].tolist()

        if pred == target:
            correct += 1

    return correct

