import torch
from Config import Config as Conf
import Config
import torch
import matplotlib.pyplot as plt

import pandas as pd
# from pandas_ml import ConfusionMatrix
import seaborn as sn
from sklearn.metrics import classification_report


def log_softmax(input):
    softmax = torch.log_softmax(input, dim=1)
    return torch.argmax(softmax, dim=1)


def convert_model_output(model_out):
    arg_max = log_softmax(model_out)
    num_of_vectors = model_out.size()[0]

    out = torch.zeros((num_of_vectors, Conf.classes))
    out[torch.arange(num_of_vectors), arg_max] = 1

    return out


def print_plot(plot_1, plot_2, labels, x_label, y_label, save=False, plot_filename=''):
    """

    :param plot_1:
    :param plot_2:
    :param labels:
    :param x_label:
    :param y_label:
    :param save:
    :param plot_filename:
    :return:
    """
    epochs = [i + 1 for i in range(len(plot_1))]

    plt.plot(epochs, plot_1, '-ok', color='red', label=labels[0])
    plt.plot(epochs, plot_2, '-ok', color='blue', label=labels[1])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title('Training results')

    if save:
        plt.savefig('varie/' + plot_filename + '.png')
    plt.show()


def print_confusion_matrix(pred_labels, true_labels):
    softmax = torch.nn.functional.log_softmax(pred_labels, dim=1)
    pred_labels = torch.argmax(softmax, dim=1)

    y_true_np = true_labels.to('cpu').numpy()
    y_pred_np = pred_labels.to('cpu').numpy()

    data = {
        'y_Actual':     y_true_np,
        'y_Predicted':  y_pred_np
    }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins=True)

    # confusion_matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])
    # confusion_matrix.print_stats()

    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    print(classification_report(y_true_np, y_pred_np, target_names=Config.class_names))

