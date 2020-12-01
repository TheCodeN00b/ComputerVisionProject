from PIL import Image
from images import JpgImageIO as jpg_utils


def clear_channel(color_channel):
    """
    The function clears the color channel, eliminating any variations
    and setting it to 0 or 255.

    :param color_channel: one of the color channel
    :return: 0 if the channel is less then 127
            1 if the channel is greater or equal then 255
    """
    if color_channel < 120:
        color_channel = 0
    else:
        color_channel = 255
    return color_channel


def clean_pixel(pixel):
    """
    The function clear the pixel and transform it in a black or white
    pixel. The function uses the CLEAR_CHANNEL function.

    :param pixel: the pixel who need to be cleaned
    :return: the cleaned pixel
    """
    red_channel = pixel[0]
    green_channel = pixel[1]
    blue_channel = pixel[2]

    red_channel = clear_channel(red_channel)
    green_channel = clear_channel(green_channel)
    blue_channel = clear_channel(blue_channel)

    cleaned_pixel = (red_channel, green_channel, blue_channel)
    return cleaned_pixel


def clean_image(image):
    """
    The function runs the whole image, calling the CLEAN_PIXEL function
    in order to transform the colored image into a black and white image
    eliminating any variations of grey or color.

    :param image: the image who need to be cleaned
    :return: the black and white image
    """

    image_width = len(image)
    image_height = len(image[0])

    cleaned_image = []

    pixels_sum = [0, 0, 0]
    c = 0

    for x in range(image_width):
        cleaned_row = []
        for y in range(image_height):
            pixels_sum[0] += image[x][y][0]
            pixels_sum[1] += image[x][y][1]
            pixels_sum[2] += image[x][y][2]
            c += 1
            cleaned_row.append(0)
        cleaned_image.append(cleaned_row)

    avg_pixel = (pixels_sum[0]//c, pixels_sum[1]//c, pixels_sum[2]//c)
    print("Average pixel before cleaning: " + str(avg_pixel))

    cord_x = 0
    for column in image:
        cord_y = 0
        for pixel in column:
            cleaned_pixel = clean_pixel(pixel)
            cleaned_image[cord_x][cord_y] = cleaned_pixel
            cord_y = cord_y + 1
        cord_x = cord_x + 1

    return cleaned_image


def resize_image_from_matrix(image):
    """
    The function resize an image from a matrix, returning a new
    matrix 28 x 28 pixels.

    :param image: the original matrix image
    :return: the resized image
    """
    # im_edit.clean_image(image)

    resized_image = Image.new("RGB", (len(image), len(image[0])), "white")
    pixels = resized_image.load()
    for x in range(len(image)):
        for y in range(len(image[0])):
            pixels[x, y] = tuple(image[x][y])

    resized_image = resized_image.resize((28, 28), Image.ANTIALIAS)
    resized_matrix = jpg_utils.convert_pillowImage_to_matrixImage(resized_image)
    return resized_matrix


def fill_non_squared_image(image):
    """
    The function transforms a non square image into a squared image.

    :param image: the original image
    :return: a squared image
    """
    squared_image = []

    original_width = len(image)
    original_height = len(image[0])

    if original_height > original_width:
        if original_height % 2 == 0:
            pixel_to_fill = (original_height - original_width) / 2
        else:
            pixel_to_fill = (original_height - original_width) // 2

        for x in range(original_height):
            for y in range(original_height):
                if x < pixel_to_fill or x >= pixel_to_fill + len(image):
                    squared_image[x][y] = (255, 255, 255)
                elif pixel_to_fill <= x < pixel_to_fill + len(image):
                    squared_image[x][y] = image[x][y]

    else:
        if original_width % 2 == 0:
            pixel_to_fill = (original_width - original_height) / 2
        else:
            pixel_to_fill = (original_width - original_height) // 2

        for x in range(original_width):
            for y in range(original_width):
                if y < pixel_to_fill or y >= pixel_to_fill + len(image[0]):
                    squared_image[x][y] = (255, 255, 255)
                elif pixel_to_fill <= y < pixel_to_fill + len(image[0]):
                    squared_image[x][y] = image[x][y]

    return squared_image
