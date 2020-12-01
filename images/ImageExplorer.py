from PIL import Image
from images import ImageEditor as im_editor
from images import JpgImageIO as im_io


def explore_image(image):
    """
    The function explores an image looking for symbols to extract.
    The threshold chosen is:

        x//28 - x//(28*2)

    where x is the max between image height and width. (No particular
    reason to choose this threshold, I just found it works well)

    :param image: the image that you want to explore
    :return: a tuple containing:
        - a list of symbols extracted (each symbol is a list of pixel tuples)
        - a tuple containing the width and height of the input image
    """

    print("Analising image...")

    cleaned_image = im_editor.clean_image(image)

    print("Image cleaned")

    cleaned_image = __clean_with_filter(cleaned_image)

    width = len(image)
    height = len(image[0])

    black = (0, 0, 0)
    white = (255, 255, 255)
    symbols_extracted = []

    for x in range(width):
        for y in range(height):
            if cleaned_image[x][y] == black:
                symbol_found = __extract_symbol(cleaned_image, x, y)
                symbols_extracted.append(symbol_found)
                for pixel in symbol_found:
                    cleaned_image[pixel[0]][pixel[1]] = white

    print("Done!")

    symbols_return = []
    print("Estratti originali: ", (len(symbols_extracted)))
    for s in symbols_extracted:
        if len(s) >= 30:
            symbols_return.append(s)

    print("Estratti filtrati: ", (len(symbols_return)))
    return symbols_return, (width, height)


def __extract_symbol(image, x, y):
    """
    This method finds black pixels that are connected to each
    other or that have distance <= threshold using the appropriate
    functions.

    :param image: the image to analise
    :param x: the x coordinate of the starting point
    :param y: the y coordinate of the starting point
    :return: a list containing the pixels of the symbol
    """

    symbol_pixels = __pixels_bfs_visit(image, x, y)

    '''
    width = len(image)
    height = len(image[0])

    mx = max(width,height)
    threshold = mx//28 - mx//(28*2)

    #print("Threshold used: "+ str(threshold))
    
    found_using_threshold = __find_by_threshold(image, symbol_pixels, threshold)

    while len(found_using_threshold) != 0:
        new_explored_pixels = set()
        for pixel in found_using_threshold:
            adjacents = __pixels_bfs_visit(image,pixel[0],pixel[1])
            for p in adjacents:
                new_explored_pixels.add(p)

        for p in new_explored_pixels:
            symbol_pixels.append(p)

        found_using_threshold = __find_by_threshold(image, symbol_pixels, threshold)
    '''
    return symbol_pixels


def __pixels_bfs_visit(image, x, y):
    """
    This method does a bfs visit of all the connected black pixels.

    :param image: the image to analise
    :param x: the x coordinate of the starting point
    :param y: the y coordinate of the starting point
    :return: a list of pixels
    """

    symbol_pixels = []
    pixels_to_visit = [(x, y)]

    while len(pixels_to_visit) != 0:
        current_pixel = pixels_to_visit[0]
        pixels_to_visit.pop(0)
        symbol_pixels.append(current_pixel)

        adjacents = __get_adjacent_pixels(image, symbol_pixels, pixels_to_visit, current_pixel[0], current_pixel[1])
        for pixel in adjacents:
            pixels_to_visit.append(pixel)

    return symbol_pixels


def __get_adjacent_pixels(image, symbol_pixels, pixels_to_visit, x, y):
    """
    This method finds the pixels adjacent to the pixel of
    coordinates given in input

    :param image: the image to analise
    :param symbol_pixels: the list of visited black pixels
    :param pixels_to_visit: the list of black pixels that need to be visited next
    :param x: the x coordinate of the starting point
    :param y: the y coordinate of the starting point
    :return: a list of pixels
    """

    adjacents = []

    if __is_inside(image, x - 1, y - 1):
        adjacents.append((x - 1, y - 1))

    if __is_inside(image, x - 1, y):
        adjacents.append((x - 1, y))

    if __is_inside(image, x - 1, y + 1):
        adjacents.append((x - 1, y + 1))

    if __is_inside(image, x, y - 1):
        adjacents.append((x, y - 1))

    if __is_inside(image, x, y + 1):
        adjacents.append((x, y + 1))

    if __is_inside(image, x + 1, y - 1):
        adjacents.append((x + 1, y - 1))

    if __is_inside(image, x + 1, y):
        adjacents.append((x + 1, y))

    if __is_inside(image, x + 1, y + 1):
        adjacents.append((x + 1, y + 1))

    black_adjacents = []
    for pixel in adjacents:
        if image[pixel[0]][pixel[1]] == (0, 0, 0) and pixel not in symbol_pixels and pixel not in pixels_to_visit:
            black_adjacents.append(pixel)

    return black_adjacents


def __is_inside(image, x, y):
    """
    This method checks if a pixel is inside an image.

    :param image: the image analised
    :param x: the x coordinate of the pixel
    :param y: the y coordinate of the pixel
    :return: True if is inside the image, False otherwise
    """

    width = len(image)
    height = len(image[0])

    return 0 <= x < width and 0 <= y < height


def __find_by_threshold(image, symbol_pixels, threshold):
    """
    This method finds all pixels that have distance <= threshold
    from the symbol that we found.

    :param image: the image analised
    :param symbol_pixels: the pixels of the symbol
    :param threshold: the threshold
    :return: the list of pixels that have distance <= threshold
    from the symbol
    """

    found = []
    for pixel in symbol_pixels:
        for x in range(-threshold, threshold + 1):
            for y in range(-threshold, threshold + 1):
                adj_pixel = (pixel[0] + x, pixel[1] + y)
                if __is_inside(image, adj_pixel[0], adj_pixel[1]):
                    if image[adj_pixel[0]][adj_pixel[1]] == (0, 0, 0):
                        if (adj_pixel[0], adj_pixel[1]) not in symbol_pixels and (
                        adj_pixel[0], adj_pixel[1]) not in found:
                            found.append((adj_pixel[0], adj_pixel[1]))

    return found


def saveSymbolsImages(symbols, image_dimensions):
    """
    This method saves the symbols found in the image exploration
    in separate images, maintaining their original position.
    :param symbols: the symbols found during the exploration
    :param image_dimensions: the size of the image that was analyzed
    """

    c = 0
    for symbol in symbols:
        im = Image.new("RGB", image_dimensions, color="white")
        pix = im.load()
        c += 1
        for pixel in symbol:
            pix[pixel[0], pixel[1]] = (0, 0, 0)
        im.save("symbol_" + str(c) + ".jpg")


def getSymbolRectangle(symbol):
    """
    This method calculates the rectangle containing the symbol given as input

    :param symbol: the list of coordinates of a symbol obtained using
    the image exploration function
    :return: a list containing two tuples, one with the x,y coordinates of the
    top left corner and the other with the x,y coordinates of the bottom right
    corner
    """

    min_x = float("inf")
    min_y = float("inf")
    max_x = 0
    max_y = 0

    for coord in symbol:
        if coord[0] < min_y:
            min_y = coord[0]
        if coord[1] < min_x:
            min_x = coord[1]
        if coord[0] > max_y:
            max_y = coord[0]
        if coord[1] > max_x:
            max_x = coord[1]

    return [(min_x, min_y), (max_x, max_y)]


def __clean_with_filter(image):
    '''
    This function takes an image as a matrix of pixels and uses a filter to
    reduce every black line it finds and then it enhances them back. This will
    help removing noise from the image while keeping the symbols intact.

    :param image: the matrix of pixels representing the image
    :return: the cleaned image
    '''

    width = len(image)
    height = len(image[0])

    max_pooling_image = [row[:] for row in image]

    # using + sign even when redundant so the formatting is nice
    positions = [(-1, -1), (+0, -1), (+1, -1),
                 (-1, +0), (+0, +0), (+1, +0),
                 (-1, +1), (+0, +1), (+1, +1)]

    for x in range(width):
        for y in range(height):
            max_pooling_image[x][y] = __max_pooling(image, x, y, positions)

    im_io.save_jpg_image("max_pooling_image.jpg", max_pooling_image)

    min_pooling_image = [row[:] for row in max_pooling_image]

    '''
    for x in range(width):
        for y in range(height):
            min_pooling_image[x][y] = __min_pooling(max_pooling_image,x,y,positions)

    im_io.save_jpg_image("min_pooling_image.jpg",min_pooling_image)
    '''

    return min_pooling_image


def __max_pooling(image, x, y, positions):
    '''
    This method calculates the value of the pixel in that position
    by taking the max pooling

    :param image: the input image
    :param x: the x coordinate of the pixel
    :param y: the y coordinate of the pixel
    :param positions: a list of positions (for commodity)
    :return: the value of the pixel
    '''

    max_pixel = image[x][y]

    for p in positions:
        if __is_inside(image, x + p[0], y + p[1]):
            found_pixel = image[x + p[0]][y + p[1]]

            if found_pixel[0] > max_pixel[0] and found_pixel[1] > max_pixel[1] and found_pixel[2] > max_pixel[2]:
                max_pixel = found_pixel

    return max_pixel


def __min_pooling(image, x, y, positions):
    '''
    This method calculates the value of the pixel in that position
    by taking the min pooling

    :param image: the input image
    :param x: the x coordinate of the pixel
    :param y: the y coordinate of the pixel
    :param positions: a list of positions (for commodity)
    :return: the value of the pixel
    '''

    min_pixel = image[x][y]

    for p in positions:
        if __is_inside(image, x + p[0], y + p[1]):
            found_pixel = image[x + p[0]][y + p[1]]

            if found_pixel[0] < min_pixel[0] and found_pixel[1] < min_pixel[1] and found_pixel[2] < min_pixel[2]:
                min_pixel = found_pixel

    return min_pixel
