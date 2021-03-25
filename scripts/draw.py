import cv2 as cv
import numpy as np


class Shape:
    def __init__(self, image, color: tuple):
        self.__image = None
        self.__image = None
        self.__color = None
        self.__thickness = None
        self.__line_type = None

        self.__set_image(image)
        self.__set_color(color)


    def __set_image(self, image):
        self.__image = image

    def __set_color(self, color):
        self.__color = color

    def __set_thickness(self, thickness):
        pass

    def __set_line_type(self, line_type):
        pass


class Circle(Shape):
    def __init__(self, image, radius: int, center: tuple, color: tuple):
        super().__init__(image, color)
        self.__radius = None
        self.__center = None
        self.__circle = None
        self.__set_radius(radius)
        self.__set_center(center)


    def __set_radius(self, radius):
        self.__radius = radius

    def __set_center(self, center):
        self.__center = center

    def update(self):
        self.__circle = cv.circle(image, self.__center, self.__radius, (50,50,50))

    def get_circle(self):
        return self.__circle


class EmptyImage:
    def __init__(self, width, height):
        self.__width = None
        self.__height = None
        self.__set_width(width)
        self.__set_height(height)

    def __set_width(self, width):
        self.__width = width

    def __set_height(self, height):
        self.__height = height

    def get_width(self):
        return self.__width

    def get_height(self):
        return self.__height


class RedBloodCell(EmptyImage):
    def __init__(self, width, height):
        super().__init__(width, height)


image = np.zeros((50, 50, 3), np.uint8)
print(image)

image = cv.circle(image, center=(25, 25), radius=20, color=(0, 0, 255), thickness=1)
cv.imwrite("test.png", image)
