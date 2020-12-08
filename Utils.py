import torch
from Config import Config as Conf
import torch
import matplotlib.pyplot as plt


def convert_model_output(model_out):
    softmax = torch.log_softmax(model_out, dim=1)
    arg_max = torch.argmax(softmax, dim=1)

    num_of_vectors = model_out.size()[0]

    out = torch.zeros((num_of_vectors, Conf.classes))
    out[torch.arange(num_of_vectors), arg_max] = 1

    return out


def print_plot_from_file(plot_1, plot_2, labels, x_label, y_label):
    epochs = [i + 1 for i in range(len(plot_1))]

    plt.plot(epochs, plot_1, '-ok', color='red', label=labels[0])
    plt.plot(epochs, plot_2, '-ok', color='blue', label=labels[1])
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title('Training results')

    plt.show()
