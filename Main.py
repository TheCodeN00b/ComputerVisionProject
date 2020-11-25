from Dataset import SymbolsDataset
from Config import Config as Conf


if __name__ == '__main__':
    print('[Main]')

    train_dataset = SymbolsDataset()
    test_dataset = SymbolsDataset(train=False)

    train_dataset.info()
