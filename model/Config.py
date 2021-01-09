from dataclasses import dataclass
from enum import Enum

import sys

import torch.nn as nn
import torch.optim as optim


class Users(Enum):
    FRANC = "franc"
    ANDREA = "andrea"
    DANIEL = "daniel"


user = Users.FRANC

symbol_to_idx = {
    '(': 0,
    ')': 1,
    '+': 2,
    '-': 3,
    '0': 4,
    '1': 5,
    '2': 6,
    '3': 7,
    '4': 8,
    '5': 9,
    '6': 10,
    '7': 11,
    '8': 12,
    '9': 13,
    'log': 14,
    'sqrt': 15,
    'x': 16
}

idx_to_symbol = {
    0: '(',
    1: ')',
    2: '+',
    3: '-',
    4: '0',
    5: '1',
    6: '2',
    7: '3',
    8: '4',
    9: '5',
    10: '6',
    11: '7',
    12: '8',
    13: '9',
    14: 'log',
    15: 'sqrt',
    16: 'x'
}


class_names = [
    '(',
    ')',
    '+',
    '-',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'log',
    'sqrt',
    'x'
]

symbol_to_idx_len = len(symbol_to_idx.keys())
idx_to_symbol_len = len(idx_to_symbol.keys())

if symbol_to_idx_len != idx_to_symbol_len:
    error_string = ''
    if symbol_to_idx_len > idx_to_symbol_len:
        error_string = 'symbol_to_idx'
    else:
        error_string = 'idx_to_symbol'
    error_string += ' has more keys'
    print('Encoding dictionaries have different number of values: ' + error_string)
    sys.exit(1)


@dataclass()
class Config:
    if user == Users.FRANC:
        train_dataset_filepath = "C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/train_data"
        test_dataset_filepath = "C:/Users/franc/OneDrive/Documents/University/Magistrale/Computer Vision/test_data"

    symbol_detector_filename = 'le_net_5.pt'
    use_cuda = True

    classes = symbol_to_idx_len
    img_size = 32
    train_dataset_size = 40 * 1000
    test_dataset_size = int(0.2 * train_dataset_size)

    # Training data
    batch_size = 16
    num_of_epochs = 40

    # model_checkpoint data
    hidden_1 = 32
    hidden_2 = 64
    hidden_3 = 128
    hidden_4 = 256
    hidden_5 = 512
    hidden_6 = 1024
    flat_layer_features = 512
