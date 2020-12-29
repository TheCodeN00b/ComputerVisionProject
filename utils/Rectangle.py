class Rectangle:
    top_left_corner = []
    top_right_corner = []
    bottom_left_corner = []
    bottom_right_corner = []
    center = []
    width = 0
    height = 0

    def __init__(self, top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner, center, width,
                 height):
        self.top_left_corner = top_left_corner
        self.top_right_corner = top_right_corner
        self.bottom_left_corner = bottom_left_corner
        self.bottom_right_corner = bottom_right_corner
        self.center = center
        self.width = width
        self.height = height