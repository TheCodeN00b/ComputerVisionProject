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

from images import JpgImageIO as JpgIO
from images import ImageExplorer
from sort import FunctionExtractions as F

from tqdm import tqdm


def run_confusion_matrix_test(test_dataset, model):
    checkpoint = torch.load('model_checkpoint/' + Conf.symbol_detector_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    predicted_labels = torch.zeros((len(test_dataset), Conf.classes)).to('cuda' if Conf.use_cuda else 'cpu')
    target_labels = torch.zeros(len(test_dataset)).to('cuda' if Conf.use_cuda else 'cpu')

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            sample, target = test_dataset[i: i + 1]

            out, _ = model(sample)
            predicted_labels[i] = out
            target_labels[i] = target

    u.print_confusion_matrix(predicted_labels, target_labels)


if __name__ == '__main__':
    print('[Main]')

    # build_bold_dataset()
    # cleanse_dataset()
    #
    # train_dataset = SymbolsDataset(Conf.train_dataset_filepath + '_bold', size=Conf.train_dataset_size)
    # train_dataset.balance_dataset()
    # train_dataset.print_info()
    #
    # test_dataset = SymbolsDataset(Conf.test_dataset_filepath + '_bold', size=Conf.test_dataset_size)
    # test_dataset.balance_dataset()
    # test_dataset.print_info()
    # #
    model = LeNet5(16)
    if Conf.use_cuda:
        model.to('cuda')

    # print(model)

    # t.train_model(
    #     train_dataset=train_dataset,
    #     test_dataset=test_dataset,
    #     model=model,
    #     loss_func=nn.CrossEntropyLoss(),
    #     optimizer=optim.Adam(lr=1e-4, params=model.parameters())
    # )

    # # sample, targets = train_dataset[0: 2]
    # # out = model_checkpoint(sample)
    # # vis_graph = make_dot(out, params={**{'inputs': sample}, **dict(model_checkpoint.named_parameters())})
    # # vis_graph.view()

    # run_confusion_matrix_test(test_dataset, model)

    # checkpoint = torch.load('model_checkpoint/' + Conf.symbol_detector_filename)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # #
    # equation_img = JpgIO.open_jpg_image('test/eq_01.jpg', bw=True)
    # symbols, tree = F.create_function(equation_img)
    # print()
    # u.print_node(tree, 0)

    equation_img = JpgIO.open_jpg_image('test/eq_06.jpg', bw=True)
    symbols, tree = F.create_function(equation_img)
    print()
    u.print_node(tree, 0)
    # equation_img = JpgIO.open_jpg_image('test/eq_05.jpg', bw=True)
    # F.create_function(equation_img)

    # symbols, _ = ImageExplorer.explore_image(equation_img)
    # len(symbols)
    #
    # for i in range(len(symbols)):
    #     symbol = symbols[i]
    #
    #     pixels = symbol[0]
    #     left_most = symbol[1]
    #     top_most = symbol[2]
    #     right_most = symbol[3]
    #     bottom_most = symbol[4]
    #     center = symbol[5]
    #
    #     width = right_most[0] - left_most[0] + 1
    #     height = bottom_most[1] - top_most[1] + 1
    #
    #     symbol_img = torch.ones((1, 1, height, width)).to('cuda')
    #     for pixel in pixels:
    #         symbol_img[0, 0, pixel[1] - top_most[1], pixel[0] - left_most[0]] = torch.Tensor([0])
    #
    #     data_transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         # transforms.ColorJitter(contrast=1000),
    #         transforms.Resize((32, 32)),
    #         transforms.ToTensor()
    #     ])
    #     out, last_conv = model(data_transform(symbol_img[0, 0].to('cpu')).view(1, 1, Conf.img_size, Conf.img_size).to('cuda'))
    #     out = out
    #     print('idx:', u.log_softmax(out)[0].tolist(), ' symbol', Config.idx_to_symbol[u.log_softmax(out)[0].tolist()])
    #
    #     symbol_img = transforms.ToPILImage(mode='L')(symbol_img[0, 0].to('cpu'))
        # symbol_img.save('test/symbol_' + str(i) + '.jpg')
        #
        # pil_img = transforms.ToPILImage(mode='L')(last_conv[0, 0].to('cpu'))
        # pil_img.save('test/symbol_conv_1_' + str(i) + '.jpg')
        # pil_img = transforms.ToPILImage(mode='L')(last_conv[0, 1].to('cpu'))
        # pil_img.save('test/symbol_conv_2_' + str(i) + '.jpg')
        # pil_img = transforms.ToPILImage(mode='L')(last_conv[0, 2].to('cpu'))
        # pil_img.save('test/symbol_conv_3_' + str(i) + '.jpg')
