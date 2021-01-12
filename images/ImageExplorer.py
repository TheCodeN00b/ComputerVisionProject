import math


def explore_image(image, threshold=3):
    """
    The function explores the image extracting symbols with a simple bfs visit on non-white pixels (using a certain
    threshold for white since I noticed the pixels aren't always perfectly (255,255,255) ). The function then aggregates
    the symbols found by looking at the column containing each symbol and being smart about recognizing symbols contained
    in each other (so it doesn't aggregate the square root with its argument) and also by not aggregating the fraction
    line with other symbols. This is essential because during cleaning of the image or during the writing itself,
    symbols may be composed by disconnected lines. The system is not perfect, but does a good enough job.

    :param image: the image to explore
    :param threshold: the threshold to use when determining adjacent pixels in the BFS visit. if set to 1 (default),
    only the directly adjacent pixels will be considered
    """

    print("Analising image...")

    height = len(image)
    width = len(image[0])

    white = (255, 255, 255)
    symbols_extracted = []

    for x in range(width):
        for y in range(height):
            if __above_white_threshold(image[y][x]):

                # extract the pixels composing the symbol
                symbol_found, leftmost, topmost, rightmost, bottommost = __extract_symbol(image, x, y, threshold)
                symbols_extracted.append(tuple((symbol_found, leftmost, topmost, rightmost, bottommost)))

                # turn that symbol white so that it won't be picked again by the pixel-by-pixel exploration of the image
                for pixel in symbol_found:
                    image[pixel[1]][pixel[0]] = white

    # run the aggregation algorithm
    symbols_extracted = __aggregate_symbol_parts(symbols_extracted)

    print("Done!")

    symbols_return = []
    print("Estratti originali: ", (len(symbols_extracted)))

    threshold = 0
    for symbol in symbols_extracted:
        threshold += len(symbol[0])

    threshold = int(threshold / len(symbols_extracted))

    threshold = int(threshold / 5)

    for s in symbols_extracted:
        if len(s[0]) >= threshold:
            symbols_return.append(s)

    print("Estratti filtrati (scartati i simboli con numero di pixel < ", threshold, "): ", (len(symbols_return)))
    return symbols_return, (width, height)


def __aggregate_symbol_parts(symbols_extracted):
    """
    This function aggregates multiple symbols into one if it thinks they belong to the same symbol. First of all
    possible candidates are symbols that are in the same "column" in the image, since we observed that usually symbols
    develop in height rather than in width (with some exceptions like square roots and fraction lines).
    The process is as follows:

        1 - calculate the center pixel for all of the symbols (this will be the center of the minimum area rectangle
        containing the symbol.
        2 - find the symbols that are in the same "column", meaning that their center is inside the column created by
        the rectangle of another symbol
        3 - exclude the symbols that are contained in other symbols, meaning that if we have symbols A and B, the
        intersection of their rectangles gives an area that's >= 80% (to avoid false negatives) of the smallest
        rectangle between A and B (this allows us to exclude square roots and their arguments)
        4 - exclude symbols that are flat (with height <= 10 pixels) that have at least two other symbols in their
        column. This allows us to exclude fraction lines, to prevent them from being merged to a symbol. Of course,
        we're assuming that the fraction line drawn is pretty straight, otherwise it won't be excluded and will end up
        merged to a symbol.
        5 - merge the symbols that have passed all the filtering of the above points and calculate the new rectangle as
        well as the new center of the merged symbol.

    :param symbols_extracted:
    :return: a list of tuples that contain:
                                        0 - list of coordinates of the pixels, in the form (x,y)
                                        1 - coordinate of the left_most pixel, in the form (x,y)
                                        2 - coordinate of the top_most pixel, in the form (x,y)
                                        3 - coordinate of the right_most pixel, in the form (x,y)
                                        4 - coordinate of the bottom_most pixel, in the form (x,y)
                                        5 - coordinate of the center pixel, in the form (x,y)
    """

    # variables for calculating the average height and width of the symbols that have been extracted
    sum_symbol_height = 0
    sum_symbol_width = 0
    number_symbols_for_avg = 0

    # list that will contain the symbols extracted with added center pixel information
    enriched_symbols_list = []

    for symbol_tuple in symbols_extracted:

        # calculated by exploiting the left_most, top_most... ecc pixels (see method doc for details)
        symbol_height = symbol_tuple[4][1] - symbol_tuple[2][1]
        symbol_width = symbol_tuple[3][0] - symbol_tuple[1][0]

        # exclude symbols that are too small from the average (this is because noise can be present in the image)
        if len(symbol_tuple[0]) >= 30:
            sum_symbol_height += symbol_height
            sum_symbol_width += symbol_width
            number_symbols_for_avg += 1

        # calculate center pixel and add it to the tuple of symbol info
        center_pixel = symbol_tuple[1][0] + int(symbol_width / 2), symbol_tuple[2][1] + int(symbol_height / 2)
        symbol_tuple_list = list(symbol_tuple)
        symbol_tuple_list.append(center_pixel)
        symbol_tuple = tuple(symbol_tuple_list)
        enriched_symbols_list.append(symbol_tuple)

    # calculate average height and width
    avg_symbol_height = int(sum_symbol_height / number_symbols_for_avg)
    avg_symbol_width = int(sum_symbol_width / number_symbols_for_avg)

    # the dictionary containing merging info. each entry is in the following structure: key: [(val, dist)...]
    # where key and val are indexes of symbols in the enriched_symbols_list and dist is the distance in pixels between
    # their center pixels.
    # EXAMPLE: if symbol 1 is in the same column with symbol 0 and 2, and the centers have distance 24 and 36 pixels
    # respectively, then merges_found[1] = [(0,24),(2,36)]
    merges_found = dict()

    # find symbols to merge, identifying them with their index in the enriched_symbols_list list created earlier
    for symbol_tuple in enriched_symbols_list:
        for compare_tuple in enriched_symbols_list:

            # avoid matching a symbol with itself
            if symbol_tuple != compare_tuple:

                # check if the center point of symbol is inside the column of the compare symbol
                if __is_center_inside_symbol_column(symbol_tuple, compare_tuple):

                    # check that the symbols are not contained in each other
                    if not __is_symbol_contained_in_symbol(symbol_tuple, compare_tuple) and not __is_symbol_contained_in_symbol(compare_tuple, symbol_tuple):

                        # populate the dictionary with merge information
                        key = enriched_symbols_list.index(symbol_tuple)
                        value = enriched_symbols_list.index(compare_tuple)
                        center_dist = __calculate_distance_from_centers(symbol_tuple, compare_tuple)
                        merges_found = __add_to_dict(merges_found, key, value, center_dist)

    return __process_dictionary(merges_found, enriched_symbols_list, avg_symbol_height)


def __process_dictionary(dictionary, symbols_extracted, avg_symbol_height):
    """
    This function processes the dictionary and does the merging of the symbols.

    :param dictionary:
    :param symbols_extracted:
    :param avg_symbol_height:
    :return: a list of symbol tuples with the processed symbols
    """

    # process the dictionary to exclude the fraction line, preventing it from being merged to a symbol by mistake
    dictionary = __find_and_exclude_fraction_line(dictionary, symbols_extracted, avg_symbol_height)

    # using a set containing tuples of symbol ids sorted"
    to_be_merged = set()
    for key in dictionary.keys():

        # for every tuple (val, dist)
        for t in dictionary[key]:

            # if the distance between the centers is less or equal to the average height of a symbol in the image
            if t[1] <= int(avg_symbol_height):

                # create and add the tuple to the set, ordering its two elements
                tuple_ids = tuple((min(key, t[0]), max(key, t[0])))
                to_be_merged.add(tuple_ids)

    symbols_to_return = []

    # create a merge_dict that contains data from to_be_merged set aggregated as follows:
    # the key of the dictionary is a symbol id; the value is a list of symbol ids, representing the ids that need to be
    # all merged together with the key
    merge_dict = {}
    for tuple_ids in to_be_merged:
        if tuple_ids[0] not in merge_dict:
            merge_dict[tuple_ids[0]] = [tuple_ids[1]]
        else:
            l = merge_dict[tuple_ids[0]]
            l.append(tuple_ids[1])
            merge_dict[tuple_ids[0]] = l

    # iterate over the merge_dict and merge all symbols
    for key in merge_dict.keys():

        # merge the symbols' pixel coordinates lists
        merged_symbol = symbols_extracted[key][0]
        for s_id in merge_dict[key]:
            merged_symbol.extend(symbols_extracted[s_id][0])

        # initialize values to default (the first pixel in the pixels coordinates list)
        left_most = merged_symbol[0]
        top_most = merged_symbol[0]
        right_most = merged_symbol[0]
        bottom_most = merged_symbol[0]

        # iterate over all of the pixels coordinates list and update the above values
        for coord in merged_symbol:
            if coord[0] < left_most[0]:
                left_most = coord
            elif coord[0] > right_most[0]:
                right_most = coord
            if coord[1] < top_most[1]:
                top_most = coord
            elif coord[1] > bottom_most[1]:
                bottom_most = coord

        # calculate the center pixel
        center_pixel = int((left_most[0] + right_most[0]) / 2), int((top_most[1] + bottom_most[1]) / 2)

        # create the symbol tuple with the computed information
        symbols_to_return.append(tuple((merged_symbol, left_most, top_most, right_most, bottom_most, center_pixel)))

    # finally, add all of the symbols that were not merged to anything to the symbols_to_return list
    for i in range(len(symbols_extracted)):
        if __index_not_in_tuple_set(i, to_be_merged):
            symbols_to_return.append(symbols_extracted[i])

    return symbols_to_return


def __find_and_exclude_fraction_line(dictionary, symbols_extracted, avg_symbol_height):
    """
    This function excludes symbols that are flat (with height <= 10 pixels) that have at least two other symbols in
    their column. This allows us to exclude fraction lines, to prevent them from being merged to a symbol. Of course,
    we're assuming that the fraction line drawn is pretty straight, otherwise it won't be excluded and will end up
    merged to a symbol.

    :param dictionary:
    :param symbols_extracted:
    :return: the dictionary modified to exclude the fraction lines
    """

    keys_to_remove = []
    for key in dictionary.keys():

        # calculate the height by delta_y of the bottom_most and top_most pixels
        symbol_height = symbols_extracted[key][4][1] - symbols_extracted[key][2][1]
        symbol_width = symbols_extracted[key][3][0] - symbols_extracted[key][1][0]

        # if it's relatively flat and has at least two symbols in its column
        if symbol_height <= avg_symbol_height / 3 and len(dictionary[key]) >= 2:

            # check if its width is greater than all of his neighbors
            is_max = True
            for val in dictionary[key]:
                other_key = val[0]
                other_width = symbols_extracted[other_key][3][0] - symbols_extracted[other_key][1][0]
                if other_width > symbol_width:
                    is_max = False

            # if it is, it's a fraction line
            if is_max:
                keys_to_remove.append(key)

    dictionary_to_return = {}
    for key in dictionary.keys():

        # if the key isn't the id of a fraction line
        if key not in keys_to_remove:
            l = []

            # exclude all tuple elements (value, dist) with value being the id of a fraction line symbol
            for t in dictionary[key]:
                if t[0] not in keys_to_remove:
                    l.append(t)
            dictionary_to_return[key] = l

    return dictionary_to_return


def __index_not_in_tuple_set(index, tuple_set):
    """
    Utility function that checks if a value is not contained inside a tuple in a given set of tuples
    :param index:
    :param tuple_set:
    :return: True if the value is not contained in the set, False otherwise
    """

    for t in tuple_set:
        if index in t:
            return False
    return True


def __calculate_distance_from_centers(symbol_tuple, compare_tuple):
    """
    This function computes the distance in pixels between two symbols by considering the distane between their center
    pixel

    :param symbol_tuple: the first symbol (remember it's a tuple of information containing pixels list, left_most pixel,
    top_most pixel etc...)
    :param compare_tuple: the second symbol (remember it's a tuple of information containing pixels list, left_most pixel,
    top_most pixel etc...)
    :return: the distance in pixels between the centers of the symbols given in input
    """

    # get the coordinates of the centers by taking them from the symbol tuple
    symbol_center = symbol_tuple[5]
    compare_center = compare_tuple[5]

    # calculate max and min x and y needed for the vertical and horizontal distances
    max_x = max(symbol_center[0], compare_center[0])
    max_y = max(symbol_center[1], compare_center[1])
    min_x = min(symbol_center[0], compare_center[0])
    min_y = min(symbol_center[1], compare_center[1])

    # calculate and return the diagonal distance using pitagora
    delta_x_2 = (max_x - min_x)**2
    delta_y_2 = (max_y - min_y)**2
    return int(math.sqrt(delta_x_2 + delta_y_2))


def __add_to_dict(dictionary, key, value, center_dist):
    """
    This functions updates the dictionary

    :param dictionary:
    :param key:
    :param value:
    :param center_dist:
    :return: the updated dictionary
    """

    # if not already present, create the list
    if key not in dictionary:
        dictionary[key] = [tuple((value, center_dist))]

    # otherwise update it by appending the new element
    else:
        l = dictionary[key]
        l.append(tuple((value, center_dist)))
        dictionary[key] = l
    return dictionary


def __is_center_inside_symbol_column(symbol_tuple, compare_tuple):
    """
    Utility function that checks if the center of a symbol is contained in the column of another symbol
    :param symbol_tuple:
    :param compare_tuple:
    :return: True if the center of symbol_tuple is contained in the column of compare_tuple. False otherwise.
    """

    compare_center_x = compare_tuple[5][0]
    right_most_x = symbol_tuple[3][0]
    left_most_x = symbol_tuple[1][0]
    return right_most_x > compare_center_x > left_most_x


def __is_symbol_contained_in_symbol(symbol_tuple, compare_tuple):
    """
    Utility function that checks if symbol_tuple is contained inside compare_tuple with a threshold of 80%
    :param symbol_tuple:
    :param compare_tuple:
    :return: True if symbol_tuple is contained inside compare_tuple. False otherwise
    """

    # get rectangle coordinates and calculate area
    left_most = symbol_tuple[1]
    top_most = symbol_tuple[2]
    right_most = symbol_tuple[3]
    bottom_most = symbol_tuple[4]
    symbol_area = (bottom_most[1] - top_most[1]) * (right_most[0] - left_most[0])

    # get rectangle coordinates and calculate area
    compare_left_most = compare_tuple[1]
    compare_top_most = compare_tuple[2]
    compare_right_most = compare_tuple[3]
    compare_bottom_most = compare_tuple[4]
    compare_area = (compare_bottom_most[1] - compare_top_most[1]) * (compare_right_most[0] - compare_left_most[0])

    # get intersection rectangle coordinates and calculate area
    x_inters = min(right_most[0], compare_right_most[0]) - max(left_most[0], compare_left_most[0])
    y_inters = min(bottom_most[1], compare_bottom_most[1]) - max(top_most[1], compare_top_most[1])
    area_inters = 0
    if x_inters > 0 and y_inters > 0:
        area_inters = x_inters * y_inters

    # return check value depending on area computed
    if area_inters == 0:
        return False
    elif area_inters / symbol_area < 0.8:
        return False
    else:
        return True


def __above_white_threshold(pixel):
    """
    Utility function that checks if a pixel is above the white threshold (200, 200, 200)
    :param pixel:
    :return: True if it's above the threshold and must be considered as a pixel part of a symbol, False otherwise.
    """

    threshold = (200, 200, 200)
    return pixel[0] < threshold[0] and pixel[1] < threshold[1] and pixel[2] < threshold[2]


def __extract_symbol(image, x, y, threshold):
    """
    This method finds non-white pixels that are connected to each other, or within "threshold" distance.

    :param image: the image to analise
    :param x: the x coordinate of the starting point
    :param y: the y coordinate of the starting point
    :return: a list containing the pixels of the symbol, the left_most, top_most, right_most and bottom_most pixels
    """

    symbol_pixels, leftmost, topmost, rightmost, bottommost = __pixels_bfs_visit(image, x, y, threshold)

    return symbol_pixels, leftmost, topmost, rightmost, bottommost


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
