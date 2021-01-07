import random
from os import listdir
import images.JpgImageIO as imio
from PIL import Image

image_count = 0


def __generate_equation(eq_string):
    equation_symbols = []

    for symbol in eq_string:
        path = ""
        if symbol == "n":
            # number
            n = random.randint(0, 9)
            path = "symbols_from_dataset/" + str(n) + "/"
        elif symbol == "o":
            # operation
            n = random.randint(0, 1)
            path = "symbols_from_dataset/+/"
            if n == 1:
                path = "symbols_from_dataset/-/"
        elif symbol == "x":
            # x
            path = "symbols_from_dataset/x/"
        elif symbol == "l":
            # log
            path = "symbols_from_dataset/log/"
        elif symbol == "s":
            # sqrt
            path = "symbols_from_dataset/sqrt/"
        elif symbol == "(":
            path = "symbols_from_dataset/(/"
        elif symbol == ")":
            path = "symbols_from_dataset/)/"

        equation_symbols.append(__get_image_symbol(path))

    __generate_equation_images(equation_symbols)


def __generate_equation_images(equation_symbols):
    bg_path = "bg/"
    backgrounds = listdir(bg_path)
    bg_i = random.randint(0, len(backgrounds) - 1)
    bg_path += backgrounds[bg_i]
    bg_image = imio.open_jpg_image(bg_path, bw=True)
    white_bg_image = Image.new("RGB", (540, 258), "white")
    pixels = white_bg_image.load()

    total_eq_width = 0
    total_eq_height = 0
    for symbol_tuple in equation_symbols:
        total_eq_width += symbol_tuple[3][0] - symbol_tuple[1][0] + 10
        symbol_height = symbol_tuple[4][1] - symbol_tuple[2][1]
        if symbol_height > total_eq_height:
            total_eq_height = symbol_height

    if total_eq_width < 480 and total_eq_height < 200:
        start_x_offset = int((540 - total_eq_width) / 2)
        start_y_offset = int((258 - total_eq_height) / 2)

        for symbol_tuple in equation_symbols:
            for coord in symbol_tuple[0]:
                x = coord[0] + start_x_offset
                y = coord[1] + start_y_offset
                color = __get_pixel_color()
                bg_image[y][x] = color
                pixels[x, y] = color
            symbol_width = symbol_tuple[3][0] - symbol_tuple[1][0]
            start_x_offset += symbol_tuple[3][0] + 10

        __save_images(bg_image, white_bg_image)

    else:
        global image_count
        print("L'equazione " + str(image_count) + " supera le dimensioni dell'immagine")


def __save_images(dirty, clean):
    global image_count
    path_dirty = "denoising_dataset/dirty/" + str(image_count) + ".jpg"
    path_clean = "denoising_dataset/clean/" + str(image_count) + ".jpg"

    clean.save(path_clean, quality=95)
    imio.save_jpg_image(path_dirty, dirty)

    image_count += 1


def __get_pixel_color():
    base = (100, 100, 100)
    i = random.randint(-20, 20)
    return tuple((base[0] + i, base[1] + i, base[2] + 1))


def __get_image_symbol(path):
    files = listdir(path)
    i = random.randint(0, len(files) - 1)
    path += files[i]
    image = imio.open_jpg_image(path, bw=True)
    original_symbol = __extract_symbol(image)
    enhanced_symbol_coords = set()
    for coord in original_symbol[0]:
        enhanced_symbol_coords.add(coord)
        for x_th in range(-1, 1):
            for y_th in range(-1, 1):
                enhanced_symbol_coords.add(tuple((coord[0] + x_th, coord[1] + y_th)))

    return list(enhanced_symbol_coords), original_symbol[1], original_symbol[2], original_symbol[3], original_symbol[4]


def __extract_symbol(image):
    """
    Iterates through the symbol image and extracts only the black pixels
    :param image:
    :return:
    """

    height = len(image)
    width = len(image[0])
    for x in range(width):
        for y in range(height):
            if __above_white_threshold(image[y][x]):
                return __pixels_bfs_visit(image, x, y, threshold=20)


def __pixels_bfs_visit(image, x, y, threshold):
    """
    This method does a bfs visit of all the connected black pixels withing threshold distance to a pixel.

    :param image: the image to analise
    :param x: the x coordinate of the starting point
    :param y: the y coordinate of the starting point
    :param threshold: the dimension of the radius used for getting the adjacents to a pixel. if it is 1, only the direct
    adjacent pixels will be considered
    :return: a list of pixels
    """

    symbol_pixels = []
    pixels_to_visit = [(x, y)]
    leftmost = (x, y)
    rightmost = (x, y)
    topmost = (x, y)
    bottommost = (x, y)

    while len(pixels_to_visit) != 0:
        current_pixel = pixels_to_visit[0]

        # update the left_most, top_most, etc... variables while the bfs is iterating over all pixels of the symbol
        if current_pixel[0] < leftmost[0]:
            leftmost = current_pixel
        elif current_pixel[0] > rightmost[0]:
            rightmost = current_pixel
        if current_pixel[1] < topmost[1]:
            topmost = current_pixel
        elif current_pixel[1] > bottommost[1]:
            bottommost = current_pixel

        pixels_to_visit.pop(0)
        symbol_pixels.append(current_pixel)

        adjacents = __get_adjacents_with_threshold(image, symbol_pixels, pixels_to_visit, current_pixel[0], current_pixel[1], threshold)
        for pixel in adjacents:
            pixels_to_visit.append(pixel)

    return symbol_pixels, leftmost, topmost, rightmost, bottommost


def __get_adjacents_with_threshold(image, symbol_pixels, pixels_to_visit, x, y, threshold):
    """
    This function computes the pixels adjacent to a given pixel up to distance threshold
    :param image: the image that's being analyzed
    :param symbol_pixels: the list of pixel coordinates already visited in the bfs
    :param pixels_to_visit: the list of pixels that need to be visited next in the bfs
    :param x: the starting x point
    :param y: the starting y point
    :param threshold: the radius of distance within which all pixels should be considered as adjacents to (x,y)
    :return: a list of adjacent pixels in the form [(x1,y1)...(xn,yn)]
    """

    adjacents = []

    for x_th in range(-threshold, threshold + 1):
        for y_th in range(-threshold, threshold + 1):
            adj_pixel = (x + x_th, y + y_th)

            # if it's not over the image boundary
            if __is_inside(image, adj_pixel[0], adj_pixel[1]):
                if __above_white_threshold(image[adj_pixel[1]][adj_pixel[0]]):

                    # if not already visited, already added as an adjacent or already added in the pixels to be visited
                    if adj_pixel not in symbol_pixels and adj_pixel not in adjacents and adj_pixel not in pixels_to_visit:
                        adjacents.append((adj_pixel[0], adj_pixel[1]))

    return adjacents


def __is_inside(image, x, y):
    """
    This method checks if a pixel is inside an image.

    :param image: the image analyzed
    :param x: the x coordinate of the pixel
    :param y: the y coordinate of the pixel
    :return: True if is inside the image, False otherwise
    """

    height = len(image)
    width = len(image[0])

    return 0 <= x < width and 0 <= y < height


def __above_white_threshold(pixel):
    """
    Utility function that checks if a pixel is above the white threshold (200, 200, 200)
    :param pixel:
    :return: True if it's above the threshold and must be considered as a pixel part of a symbol, False otherwise.
    """

    threshold = (200, 200, 200)
    return pixel[0] < threshold[0] and pixel[1] < threshold[1] and pixel[2] < threshold[2]


if __name__ == "__main__":
    '''
    Use the eq_str inside a loop to ask the system to generate images. Images will be saved with a filename "n.jpg"
    with n being a number increasing from 0. Images will be saved in two folders (clean and dirty) with the same name.
    
    How to use the string:
        - n: number random from 0 to 9
        - o: random operation (either - or +)
        - x: variable x
        - l: log
        - ): closing bracket )
        - (: opening bracket (
        - s: sqrt
    '''
    for _ in range(3):
        eq_str = "nonoxl(x)"
        __generate_equation(eq_str)
