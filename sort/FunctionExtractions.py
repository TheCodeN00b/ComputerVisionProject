from images import ImageExplorer as ImExp
import math
from utils import EquationTreeNode as EqTree
from utils import Rectangle as Rect
import progressbar
from model.Model import Conv2DSymbolDetector
from images import JpgImageIO as jpg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch as torch

from model.Config import Config as Conf

symbols_info = []
symbols_names = []
symbols_rectangles = []
image_symbols = []


def create_function(image):
    """
    The function takes the symbols extracted and generate an ordered list which
    show the order of operations.
    :param image: the original cleaned image
    :return: a tuple (image_symbols, info) where image is a list of symbols extracted from image and info a tuple which
    show information about tuple
    """
    # Extraxts all the symbols
    global image_symbols
    image_symbols = ImExp.explore_image(image)[0]

    # For each symbol, extracts the rectangle
    bar = progressbar.ProgressBar(prefix='Extracting rectangles ', suffix=' Complete', maxval=len(image_symbols),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()

    for i in range(len(image_symbols)):
        symbol = image_symbols[i]
        symbols_rectangles.append(ImExp.getSymbolRectangle(symbol))
        bar.update(i + 1)
    bar.finish()

    # Computes the rectangle's info
    bar = progressbar.ProgressBar(prefix='Extracting information ', suffix=' Complete', maxval=len(symbols_rectangles),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()

    for i in range(len(symbols_rectangles)):
        coordinates = symbols_rectangles[i]
        rect_info = __extract_rect_information(coordinates)
        symbols_info.append(rect_info)
        bar.update(i + 1)
    bar.finish()

    print()

    # model initialization
    model = Conv2DSymbolDetector()
    checkpoint = torch.load(Conf.symbol_detector_filename)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Determine the correct math symbol with the NN
    # and put them into the symbols_name list
    for i in range(len(image_symbols)):
        symbol = __shift_symbol_coordinates(image_symbols[i])
        info = symbols_info[i]

        # safety padding: since the width and height are calculated without float division, it can
        # give an incorrect result. so just to avoid trouble, we put a padding there avoiding the
        # index out of bound
        width = info.width + 5
        height = info.height + 5
        symbol_img = Image.new("RGB", (width, height), "white")
        symbol_img_pixels = symbol_img.load()

        for j in range(len(symbol)):
            x_coord = symbol[j][1]
            y_coord = symbol[j][0]
            symbol_img_pixels[x_coord, y_coord] = 0

        # symbol_img.show()

        ratio = height / width
        print('Symbol info\n-widht:', width, '\n-height:', height, '\n-ratio', ratio)

        if ratio < 1.0:
            if ratio <= 0.15:
                new_width = 48
            else:
                new_width = height

            reshaped_symbol_img = Image.new("RGB", (new_width, height), "white")
            reshaped_symbol_img_pixels = reshaped_symbol_img.load()

            for x in range(new_width):
                for y in range(height):
                    reshaped_symbol_img_pixels[x, y] = symbol_img_pixels[x, y]

            symbol_img = reshaped_symbol_img

        resize_image = jpg.resize_image(symbol_img, 48)
        # resize_image.show()

        resize_img_matrix = jpg.convert_pillowImage_to_BW_matrixImage(resize_image)
        pixels = []
        for x in range(48):
            for y in range(48):
                pixels.append(resize_img_matrix[x][y][0])
        pixels = np.array(pixels)
        pixels = np.reshape(pixels, (48, 48))

        # resize_image.show()
        symbol_name = model.predict_image(torch.from_numpy(pixels))
        print('Symbol prediction completed\n')
        if symbol_name == '-':
            if width > len(image) / 2:
                symbol_name = 'div'
        symbols_names.append(symbol_name)

    preprocess_equation_symbols()

    __sort_equation_symbols()
    print(symbols_names)
    tree, index = __create_equation_node(len(symbols_names) - 1)

    print("Sorting complete")
    return image_symbols, tree


def aggregate_two_digits(symbol: str, symbol_info: Rect.Rectangle, symbol_idx: int, numbers_string: [str]) -> (
        str, Rect.Rectangle):
    """
    :param symbol:
    :param symbol_info:
    :param symbol_idx:
    :param numbers_string:
    :return:
    """
    global symbols_names, symbols_info

    rightmost_symbol_idx = __search_right_closest_symbol(symbols_info[symbol_idx], symbol_idx)

    if rightmost_symbol_idx != -1:
        # we have found the rightmost symbol
        rightmost_symbol = symbols_names[rightmost_symbol_idx]
        rightmost_symbol_info = symbols_info[rightmost_symbol_idx]

        # we verify that it is a digit
        if rightmost_symbol in numbers_string:
            # we aggregate the current symbol with the new digit
            symbol += rightmost_symbol
            symbol_info.bottom_right_corner = rightmost_symbol_info.bottom_right_corner
            symbol_info.top_right_corner = rightmost_symbol_info.top_right_corner

            symbols_names[rightmost_symbol_idx] = 'aggr'  # we label the rightmost symbol as aggregated

            new_symbol = aggregate_two_digits(symbol, symbols_info, rightmost_symbol_idx, numbers_string)

            if new_symbol != 'none':
                return new_symbol
            else:
                return symbol, symbol_info
        else:
            return 'op', None
    else:
        return 'none', None


def preprocess_equation_symbols():
    """
    :return: None
    """
    global symbols_names, symbols_info

    processed_symbols_names = []
    processes_symbols_info = []

    visited_symbols = []

    print('Original symbols', symbols_names)

    # the list of digits between 0 and 9 in string format
    numbers_string = [str(i) for i in range(10)]

    for symbol_idx in range(len(symbols_names)):
        symbol = symbols_names[symbol_idx]  # we read the symbol
        symbol_info = symbols_info[symbol_idx]

        # we verify that is not already aggregated
        if symbol != 'aggr':
            if symbol in numbers_string:
                new_symbol, new_symbol_info = aggregate_two_digits(symbol, symbols_info, symbol_idx, numbers_string)
                print('Aggregated symbol:', new_symbol)

                # if the function has returned a new symbol we insert it, otherwise we insert the previous one
                if new_symbol != 'none' and new_symbol != 'op':
                    processed_symbols_names.append(new_symbol)
                    processes_symbols_info.append(new_symbol_info)

                elif new_symbol == 'op':
                    # we found an operator
                    operator_idx = __search_right_closest_symbol(symbols_info[symbol_idx], symbol_idx)
                    operator = symbols_names[operator_idx]
                    operator_info = symbols_info[operator_idx]

                    if operator in ['x', 'sqrt', 'div']:
                        processed_symbols_names.append(symbol)
                        processes_symbols_info.append(symbol_info)
                        visited_symbols.append(symbol_idx)

                        processed_symbols_names.append('*')

                        mul_top_left_corner = [((i + j) // 2) - 1 for i, j in
                                               zip(operator_info.top_left_corner, symbol_info.top_right_corner)]
                        mul_top_right_corner = [((i + j) // 2) + 1 for i, j in
                                                zip(operator_info.top_left_corner, symbol_info.top_right_corner)]
                        mul_bottom_left_corner = [((i + j) // 2) - 1 for i, j in
                                                  zip(operator_info.bottom_left_corner,
                                                      symbol_info.bottom_right_corner)]
                        mul_bottom_right_corner = [((i + j) // 2) + 1 for i, j in
                                                   zip(operator_info.bottom_left_corner,
                                                       symbol_info.bottom_right_corner)]
                        mul_center = [((i + j) // 2) for i, j in
                                      zip(operator_info.center,
                                          symbol_info.center)]

                        mul_info = Rect.Rectangle(mul_top_left_corner, mul_top_right_corner,
                                                  mul_bottom_left_corner, mul_bottom_right_corner, mul_center, 2, 2)
                        processes_symbols_info.append(mul_info)

                        if operator_idx not in visited_symbols:
                            processed_symbols_names.append(operator)
                            processes_symbols_info.append(operator_info)
                            visited_symbols.append(operator_idx)

                        symbols_names[operator_idx] = 'aggr'
                    else:
                        if symbol_idx not in visited_symbols:
                            processed_symbols_names.append(symbol)
                            processes_symbols_info.append(symbol_info)
                            visited_symbols.append(symbol_idx)
                else:
                    if symbol_idx not in visited_symbols:
                        processed_symbols_names.append(symbol)
                        processes_symbols_info.append(symbol_info)
                        visited_symbols.append(symbol_idx)
            else:
                if symbol_idx not in visited_symbols:
                    processed_symbols_names.append(symbol)
                    processes_symbols_info.append(symbol_info)
                    visited_symbols.append(symbol_idx)
        else:
            print('Aggregated symbol:', symbol)

    print('Processed symbols:', processed_symbols_names)
    symbols_names = processed_symbols_names
    symbols_info = processes_symbols_info


def __shift_symbol_coordinates(symbol):
    new_symbol = []
    leftmost_pixel = __find_leftmost_pixel(symbol)
    topmost_pixel = symbol[0]
    for pixel in symbol:
        new_symbol.append((pixel[0] - topmost_pixel[0], pixel[1] - leftmost_pixel[1]))
    return new_symbol


def __find_leftmost_pixel(symbol):
    leftmost = (float("inf"), float("inf"))
    for pixel in symbol:
        if pixel[1] < leftmost[1]:
            leftmost = pixel
    return leftmost


def __symbols_are_close_in_equation(symbol_1_indices, symbol_2_indices):
    """
    The function verify if almost two indices in the input lists are
    close.
    :param symbol_1_indices: the list of indices of the first symbol
    :param symbol_2_indices: the list of indices of the second symbol
    :return: a tuple (x, y) where x is the index of the left-most symbol and y the index
    of the right-most symbol. If no exists such symbols returns (-1, -1)
    """

    for index_1 in symbol_1_indices:
        for index_2 in symbol_2_indices:
            if index_1 == index_2 + 1:
                return index_2, index_1
            elif index_2 == index_1 + 1:
                return index_1, index_2
    return -1, -1


def __search_right_closest_symbol(symbol, index):
    """
    The function search the closest math symbol to input symbol
    on the right side and returns his index.
    :param symbol: the math symbol
    :param index: math symbol's index
    :return: index of the closest symbol on the right
    """
    closest_right_rect = -1
    min_dist = float("inf")

    for index_2 in range(len(symbols_info)):
        # check if the two symbols are different
        if index_2 != index:
            # take the rect info
            rect_info = symbols_info[index_2]
            # checks if they are on the same line
            same_line = __check_if_rects_are_on_same_height(symbol, rect_info)

            # if are on the same line and the rect are on the left
            rect_center_x_coord = rect_info.center[0]
            symbol_center_x_coord = symbol.center[0]

            # if are on the same line and the rect are on the right
            if same_line and symbol_center_x_coord < rect_center_x_coord:
                leftmost_side_center_x = rect_info.bottom_left_corner[0]
                leftmost_side_center_y = rect_info.bottom_left_corner[1] - rect_info.top_left_corner[1]

                dist = math.hypot(symbol.center[0] - leftmost_side_center_x,
                                  symbol.center[1] - leftmost_side_center_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_right_rect = index_2

    return closest_right_rect


def __search_left_closest_symbol(symbol, index):
    """
    The function search the closest symbol to symbol on the left side and
    returns his index.
    :param symbol: the math symbol
    :param index: the math symbol index
    :return: the index of the closest symbol on the left side
    """
    closest_left_index = -1
    min_dist = float("inf")

    for index_2 in range(len(symbols_info)):
        # check if the two symbols are different
        if index_2 != index:
            # take the rect info
            rect_info = symbols_info[index_2]
            # checks if they are on the same line
            same_line = __check_if_rects_are_on_same_height(symbol, rect_info)

            # if are on the same line and the rect are on the left
            rect_center_x_coord = rect_info.center[0]
            symbol_center_x_coord = symbol.center[0]

            if same_line and rect_center_x_coord < symbol_center_x_coord:
                rightmost_corner_center_x = rect_info.bottom_right_corner[0]
                rightmost_corner_center_y = rect_info.bottom_right_corner[1] - rect_info.top_right_corner[1]
                dist = math.hypot(symbol.center[0] - rightmost_corner_center_x,
                                  symbol.center[1] - rightmost_corner_center_y)

                if dist < min_dist:
                    min_dist = dist
                    closest_left_index = index_2

    return closest_left_index


def __check_if_symbol_is_over(rect1, rect2):
    """
    The function verify if the first rectangle is over the second one
    :param rect1: the first rectangle
    :param rect2: the second rectangle
    :return: True if rect1 is over the second one, false otherwise
    """

    rect_center_x_coord = rect1[4][0]
    rect2_center_x_coord = rect2[4][0]
    rect2_width = rect2[5]
    rect1_center_y_coord = rect1[4][1]
    rect2_center_y_coord = rect2[4][1]

    leftmost_x_coord = rect2_center_x_coord - (rect2_width // 2)
    rightmost_y_coord = rect2_center_x_coord + (rect2_width // 2)
    if (
            leftmost_x_coord <= rect_center_x_coord <= rightmost_y_coord
            and
            rect1_center_y_coord < rect2_center_y_coord
    ):
        return True
    else:
        return False


def __check_if_rects_are_on_same_height(rect1, rect2):
    rect2_center_y_coord = rect2.center[1]
    rect2_height = rect2.height
    rect1_center_y_coord = rect1.center[1]
    rect1_height = rect1.height
    if (
            rect2_center_y_coord - (rect2_height // 2) <= rect1_center_y_coord <= rect2_center_y_coord + (
            rect2_height // 2) or
            rect1_center_y_coord - (rect1_height // 2) <= rect2_center_y_coord <= rect1_center_y_coord + (
            rect1_height // 2)
    ):
        return True
    else:
        return False


def __find_closest_symbol(symbol):
    """
    The function searches and return the closest symbol to symbol
    :param symbol:
    :return:
    """
    closest_symbol = []
    min_dist = float("inf")

    for symbol_2 in symbols_info:
        if symbol != symbol_2:
            dist = math.sqrt(
                math.pow(symbol[4][0] - symbol_2[4][0], 2) +
                math.pow(symbol[4][1] - symbol_2[4][1], 2)
            )
            if dist < min_dist:
                min_dist = dist
                closest_symbol = symbol_2

    return closest_symbol


def __check_if_symbol_is_base(symbol_1, symbol_2):
    """
    The function verify if the first symbol is the base of the second one.
    :param symbol_1: the first symbol rectangle info
    :param symbol_2: the second symbol rectangle info
    :return: True if is a base, false otherwise
    """
    is_smaller = (symbol_2[5] // 2 <= symbol_1[5] and symbol_2[6] // 2 <= symbol_1[6])
    is_not_on_the_same_height = not __check_if_rects_are_on_same_height(symbol_1, symbol_2)
    is_the_closest_one = symbol_1 == __find_closest_symbol(symbol_2)

    return is_smaller and is_not_on_the_same_height and is_the_closest_one


def __is_operand_of(operand, operator):
    if symbols_names[operator] == 'sqrt':
        return __is_contained_in(symbols_info[operand], symbols_info[operator])

    if symbols_names[operator] in ['+', '-', '*']:
        closest_left = __search_left_closest_symbol(symbols_info[operator], operator)
        closest_right = __search_right_closest_symbol(symbols_info[operator], operator)

        # In this case we verify that, if the operator is an addition and the operand is
        # involved in a multiplication, then the operator is not an operator
        if symbols_names[operator] in ['+', '-'] and operand == closest_right:
            symbol_at_right = __search_right_closest_symbol(symbols_info[closest_right], closest_right)
            if symbol_at_right != -1:
                return symbols_names[symbol_at_right] != '*'
            else:
                return True

        # same as above, but in the other side
        if symbols_names[operator] in ['+', '-'] and operand == closest_left:
            symbol_at_left = __search_left_closest_symbol(symbols_info[closest_right], closest_right)
            if symbol_at_left != -1:
                return symbols_names[symbol_at_left] != '*'
            else:
                return True

        # we verify if an addition is operator of mul
        if symbols_names[operator] in ['+', '-'] and symbols_names[operand] == '*':
            at_left_of_closest_left = __search_left_closest_symbol(symbols_info[closest_left], closest_left)
            at_right_of_closest_right = __search_right_closest_symbol(symbols_info[closest_right], closest_right)

            return operand == at_left_of_closest_left or operand == at_right_of_closest_right

        # in this case we manage the order between + and - ops
        if symbols_names[operator] in ['+', '-'] and symbols_names[operand] in ['+', '-']:
            # we consider a +(-) operation operand of another +(-) operation if the first is at left
            # respect the second one
            operator_left_element = __search_left_closest_symbol(symbols_info[operator], operator)
            if operator_left_element != -1:
                operator_left_left_element = __search_left_closest_symbol(symbols_info[operator_left_element], operator_left_element)
                return operand == operator_left_left_element

        # we verify if the operand is involved in a multiplication
        if symbols_names[operator] == '*':
            return operand == closest_left or operand == closest_right

    if symbols_names[operator] == 'div':
        return True


def __swap_and_shift(first, last):
    # temp_symbol = image_symbols[last]
    temp_info = symbols_info[last]
    temp_name = symbols_names[last]

    i = last
    while i > first:
        # image_symbols[i] = image_symbols[i - 1]
        symbols_info[i] = symbols_info[i - 1]
        symbols_names[i] = symbols_names[i - 1]
        i = i - 1

    # image_symbols[first] = temp_symbol
    symbols_info[first] = temp_info
    symbols_names[first] = temp_name


def __create_equation_node(node_index):
    root = EqTree.EquationTreeNode(None, symbols_names[node_index], [])
    children = []

    last_index = node_index - 1
    child_index = node_index - 1

    while child_index >= 0:
        if __is_operand_of(child_index, node_index):
            child, last_index = __create_equation_node(child_index)
            child.father = root
            children.append(child)
            child_index = last_index
        else:
            child_index -= 1

    root.children = children
    return root, last_index


def __sort_equation_symbols():
    global image_symbols

    print("Pre-processed symbols:", symbols_names)

    for i in range(len(symbols_names)):
        if symbols_names[i] in ['+', '-', 'sqrt', 'div', '*']:
            __swap_and_shift(0, i)

    print("Post-processed symbols:", symbols_names)

    change = True
    while change:
        change = False

        for i in range(len(symbols_names) - 1):
            _next = i
            for j in range(i + 1, len(symbols_names)):
                if __is_operand_of(j, _next):
                    __swap_and_shift(_next, j)
                    _next = _next + 1
                    change = True


def __is_less_then(i, j):
    contained_in = __is_contained_in(symbols_info[i], symbols_info[j])
    over = __check_if_symbol_is_over(symbols_info[i], symbols_info[j])
    closest_left = (i == __search_left_closest_symbol(symbols_info[j], j))

    return contained_in or over or closest_left


def __extract_rect_information(coordinates):
    """
    The function uses the coordinates of the rectangle to extract
    useful information about it.
    :param coordinates: a tuple (x, y) which specify the coordinate of the rectangle
    :return: a tuple
    (top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, center, width, height)
    """
    top_left = coordinates[0]
    bottom_right = coordinates[1]
    # take the x from the bottom_right and the y from the top_left
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])

    rect_width = top_right[0] - top_left[0]
    rect_height = bottom_right[1] - top_right[1]

    center = (rect_width // 2 + top_left[0], rect_height // 2 + top_left[1])

    return Rect.Rectangle(top_left, top_right, bottom_left, bottom_right, center, rect_width, rect_height)


def __is_contained_in(first_symbol, second_symbol):
    """
    The function examines the information of the two symbols and verify
    if the first symbol is contained into the second one. In detail, it verifies
    that the center of the first symbol is contained into the rectangle of the second and then
    verify if the size of the first one is less or equal than the second one's size.
    :param first_symbol: information tuple of the first symbol
    :param second_symbol: information tuple of the second symbol
    :return: True if the first symbol is contained into the second, False otherwise
    """

    first_symbol_top_left = first_symbol.top_left_corner
    first_symbol_top_right = first_symbol.top_right_corner
    first_symbol_bottom_left = first_symbol.bottom_left_corner
    first_symbol_bottom_right = first_symbol.bottom_right_corner

    second_symbol_top_left = second_symbol.top_left_corner
    second_symbol_top_right = second_symbol.top_right_corner
    second_symbol_bottom_left = second_symbol.bottom_left_corner
    second_symbol_bottom_right = second_symbol.bottom_right_corner

    if (
            second_symbol_top_left[0] <= first_symbol_top_left[0] and
            first_symbol_top_right[0] <= second_symbol_top_right[0] and
            second_symbol_bottom_left[0] <= first_symbol_bottom_left[0] and
            first_symbol_bottom_right[0] <= second_symbol_bottom_right[0] and

            second_symbol_top_left[1] <= first_symbol_top_left[1] and
            first_symbol_bottom_left[1] <= second_symbol_bottom_left[1] and
            second_symbol_top_right[1] <= first_symbol_top_right[1] and
            first_symbol_bottom_right[1] <= second_symbol_bottom_right[1]
    ):
        return True
    else:
        return False