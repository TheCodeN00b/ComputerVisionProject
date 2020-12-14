from PIL import Image
from PIL import ImageOps


def open_and_resize_jpg_image(filepath, size):
    """
    The function opens and resizes an image, returning the matrix of the pixels.
    :param filepath: the image's filepath
    :param size: a tuple containing height and width
    :return: the pixel's matrix
    """

    image = Image.open(filepath)
    image.thumbnail(size)
    image = image.convert("L")

    desired_size = size[0]

    if image.size != size:
        delta_w = desired_size - image.size[0]
        delta_h = desired_size - image.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        padd_img = ImageOps.expand(image, padding, fill=255)
        image = padd_img

    return convert_pillowImage_to_BW_matrixImage(image)


def resize_image(image, desired_size):
    image.thumbnail((desired_size, desired_size))
    image = image.convert("L")

    delta_w = desired_size - image.size[0]
    delta_h = desired_size - image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    padd_img = ImageOps.expand(image, padding, fill=255)
    image = padd_img

    return image


def open_jpg_image(filepath, bw):
    """
    The function opens an image, located in the specified filepath, and
    returns the matrix of the pixels.

    :param filepath: the image's filepath
    :param bw: a boolean that if set to True opens the image and converts it to black and white
    :return: the pixel's matrix, each pixel being a tuple, in a row-column order
    """
    image = Image.open(filepath)

    if bw:
        image = image.convert('L')
        return convert_pillowImage_to_BW_matrixImage(image)
    else:
        image = image.convert('RGB')
        return convert_pillowImage_to_matrixImage(image)


def save_jpg_image(filepath, pixels_matrix):
    """
    The function saves a matrix of pixels as a new image, in the
    specified filepath
    :param filepath: the output image's filepath
    :param pixels_matrix: the matrix which contains the pixels
    :param width: the width of the image
    :param height: the height of the image
    :return: nothing
    """
    height = len(pixels_matrix)
    width = len(pixels_matrix[0])

    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            pixel = pixels_matrix[y][x]
            pixels[x, y] = pixel

    image.save(filepath, quality=95)


def get_image_from_matrix(pixels_matrix):
    height = len(pixels_matrix)
    width = len(pixels_matrix[0])

    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            pixel = pixels_matrix[y][x]
            pixels[x, y] = pixel

    return image


def convert_pillowImage_to_matrixImage(pillowImage):
    """
    The function takes an image opened using Image.open() function and
    converts it to a matrix of pixels.
    (Can be a useful function, that's why I separated it from the
    open_jpg_image function)

    :param pillowImage: the image opened with pillow
    :return: the matrix representation of the image
    """

    pixels = pillowImage.load()

    image_width = pillowImage.size[1]
    image_height = pillowImage.size[0]

    imported_image = []
    for x in range(image_width):
        imported_col = []
        for y in range(image_height):
            imported_col.append(tuple(pixels[y, x]))
        imported_image.append(imported_col)

    return imported_image


def convert_pillowImage_to_BW_matrixImage(pillowImage):
    """
    The function takes an image opened using Image.open() function and
    converts it to a matrix of pixels.
    (Can be a useful function, that's why I separated it from the
    open_jpg_image function)

    :param pillowImage: the image opened with pillow
    :return: the matrix representation of the image
    """

    pixels = pillowImage.load()

    image_width = pillowImage.size[0]
    image_height = pillowImage.size[1]

    imported_image = []
    for x in range(image_width):
        imported_col = []
        for y in range(image_height):
            imported_col.append(tuple([pixels[y, x],pixels[y, x],pixels[y, x]]))
        imported_image.append(imported_col)

    return imported_image
