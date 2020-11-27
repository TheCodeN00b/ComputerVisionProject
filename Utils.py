import torch
from Config import Config as Conf


def convert_model_output(model_out):
    arg_max = torch.argmax(model_out, dim=1)

    num_of_vectors = model_out.size()[0]

    out = torch.zeros((num_of_vectors, Conf.classes))
    out[torch.arange(num_of_vectors), arg_max] = 1

    return out
