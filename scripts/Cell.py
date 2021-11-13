import json
import os.path
import random
import numpy as np
import torch
import cv2 as cv
import imageio
import scripts.methods

"""
Cell class used to track individual cells throughout videos
"""


def extractCrop(frame, center=(50, 50), cropSize=128):
    halfCrop = int(cropSize / 2)
    frameSize = frame.shape
    crop = np.zeros((cropSize, cropSize))
    # These values are to get the frame indexi
    left = max(center[1] - halfCrop, 0)
    right = min(frameSize[1], center[1] + halfCrop)
    top = max(center[0] - halfCrop, 0)
    bottom = min(frameSize[0], center[0] + halfCrop)
    # These values are for the crop
    cLeft = left + halfCrop - center[1]
    cRight = cropSize - (center[1] + halfCrop - right)
    cTop = top + halfCrop - center[0]
    cBottom = cropSize - (center[0] + halfCrop - bottom)

    crop[cTop:cBottom, cLeft:cRight] = frame[top:bottom, left:right]

    return crop


class Cell:
    def __init__(self, coordinates: np.ndarray, video_type: str, median_background: torch.tensor,
                 initial_frame_num: int):
        self.__coordinates = []
        self.__coordinates.append(self.__check_coordinates(coordinates, True))
        self.__id = random.random() * 10000
        self.__video_type = video_type
        self.__background = median_background
        self.__median_x_coordinate_end_border = self.__calculate_median_x_coordinate_border_end()
        self.__speed = 45
        self.__direction = None
        self.__initial_coordinate_offset = 10
        self.__prediction = None
        self.__set_prediction(
            np.array((self.__coordinates[-1][0], self.__coordinates[-1][1] + self.__initial_coordinate_offset)))
        self.__coordinate_predictions = []
        self.__coordinate_predictions.append(self.__prediction)
        self.__prediction_images = []
        self.__alive = True
        self.__arrived = False
        random.seed(coordinates.sum() + initial_frame_num)
        self.__color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__segmentation_crop_collection = []
        self.__frame_crop_collection = []
        self.__clean_background_crop_collection = []
        self.__completion_id = None

    def extract_segmentation(self, segmentation: np.ndarray, frame: np.ndarray):
        """
        For each cell movement, extract from the segmentation frame, real frame, and median frame, a cropped image around the cell in question
        :param segmentation: Segmentation frame from neural network
        :param frame: Real frame
        """
        frame = frame.copy()
        segmentation = segmentation.copy()
        offset = 64
        desired_shape = (offset * 2, offset * 2)
        current_coordinate = self.get_current_coordinate()
        y = current_coordinate[0]
        x = current_coordinate[1]

        segmentation_crop = extractCrop(segmentation[0, :, :], (y, x))
        frame_crop = extractCrop(frame[:, :, 0], (y, x))
        clean_background_crop = extractCrop(self.__background.detach().numpy()[0, :, :], (y, x))

        if segmentation_crop.shape == desired_shape and frame_crop.shape == desired_shape and clean_background_crop.shape == desired_shape:
            self.__segmentation_crop_collection.append(torch.tensor(segmentation_crop))
            self.__frame_crop_collection.append(torch.tensor(frame_crop))
            self.__clean_background_crop_collection.append(torch.tensor(clean_background_crop))

    def make_journey_collage(self, path_prefix=""):
        """
        Concat all frame crops and segmentation crops and save image
        :param path_prefix: Path prefix for filename
        """
        path = "{}{}/".format(path_prefix, self.get_video_type())
        frame_crops = torch.cat([item for item in self.__frame_crop_collection], 1)
        segmentation_crops = torch.cat([item for item in self.__segmentation_crop_collection], 1)
        background_crop = torch.cat([item * 255 for item in self.__clean_background_crop_collection], 1)
        combined_image = torch.cat((frame_crops, background_crop, segmentation_crops), 0)
        cv.imwrite("{}cell_journey_{}.png".format(path, self.get_completion_id()),
                   combined_image.detach().numpy())
        del self.__segmentation_crop_collection, self.__frame_crop_collection, self.__clean_background_crop_collection
        self.kill()

    def __predict_next_coordinates(self):
        """
        Using cell momentum, predict next cell coordinates
        :return: Predicted coordinates
        """
        if len(self.get_coordinate_list()) > 1:
            previous_coordinates = self.get_coordinate_list()[-2]
            current_coordinates = self.get_coordinate_list()[-1]
            dy = current_coordinates[0] - previous_coordinates[0]
            dx = current_coordinates[1] - previous_coordinates[1]
            self.__set_current_speed(dx)
            prediction = (current_coordinates[0] + dy, current_coordinates[1] + dx)
            return np.array(prediction)

    def update(self, coordinates):
        """
        Predict and update the cell object with coordinates
        :param coordinates: Sanity checked coordinates
        """
        if self.__check_coordinates(coordinates) is not None:
            self.__coordinates.append(self.__check_coordinates(coordinates))
            prediction = self.__predict_next_coordinates()
            self.__set_prediction(prediction)
            self.__coordinate_predictions.append(self.get_prediction())

    def __set_prediction(self, prediction):
        self.__prediction = prediction

    def get_prediction(self):
        return self.__prediction

    def get_id(self):
        return self.__id

    def get_coordinate_list(self):
        return self.__coordinates

    def get_current_coordinate(self):
        return self.get_coordinate_list()[-1]

    def __check_coordinates(self, coordinates, initial=False):
        """
        Sanity check for the coordinates
        :param coordinates: Array of two items
        :param initial: Indicates whether this is the first time entering a coordinate
        """
        coordinates = np.array(coordinates)

        assert type(coordinates) is np.ndarray
        if len(coordinates) != 2:
            raise Exception("Coordinates not in correct format.")
        if initial is False:
            if coordinates[1] <= self.get_coordinate_list()[-1][1]:
                return None
        return coordinates

    def draw_personal_prediction(self, frame):
        """
        Draws for a single cell a circle around the current coordinate and draws a circle for the predicted coordinate with the radius being the momentum of the cell
        :param frame: Frame to draw circles on
        """
        current_coordinate = tuple(self.get_coordinate_list()[-1][::-1])
        predicted_coordinate = tuple(self.get_prediction()[::-1])
        radius = 30

        frame = frame.copy()
        frame = cv.circle(img=frame, center=current_coordinate, radius=radius, color=self.__color,
                          thickness=2)
        frame = cv.circle(img=frame, center=predicted_coordinate, radius=radius - 15, color=self.__color,
                          thickness=1)
        self.__prediction_images.append(frame)

    def draw_prediction(self, frame):
        """
        Draws a circle around the current coordinate and draws a circle for the predicted coordinate with the radius being the momentum of the cell
        :param frame: Frame to draw circles on
        """
        current_coordinate = tuple(self.get_coordinate_list()[-1][::-1])
        predicted_coordinate = tuple(self.get_prediction()[::-1])
        radius = 30

        frame = cv.circle(img=frame, center=current_coordinate, radius=radius, color=self.__color,
                          thickness=2)
        frame = cv.circle(img=frame, center=predicted_coordinate, radius=self.__speed, color=self.__color,
                          thickness=1)

    def is_alive(self):
        return self.__alive

    def kill(self):
        self.__alive = False

    def arrived(self, frame_num: int):
        """
        Cells that have arrived get a completion id that has the frame number and last prediction in it
        :param frame_num: Frame number where cell completion happened
        """
        self.__arrived = True
        prediction = [str(yx) for yx in self.get_prediction()]
        self.__completion_id = "{}_{}_{}".format(self.get_video_type(), str(frame_num),
                                                 ";".join(prediction))

    def __calculate_deformity_metrics(self, img, threshold):
        """
        Calculates the deformity index and deformity ratio for segmentation images
        :param img: Segmentation image (binary)
        :param threshold: Color threshold
        :return: Deformity metrics
        """
        ret, thresh = cv.threshold(img, threshold, 255, 0)
        contours, hierarchy = cv.findContours(thresh, 1, 2)
        cnt = contours[0]
        boxes = []
        for c in cnt:
            (x, y, w, h) = cv.boundingRect(c)
            boxes.append([x, y, x + w, y + h])

        boxes = np.asarray(boxes)
        left, top = np.min(boxes, axis=0)[:2]
        right, bottom = np.max(boxes, axis=0)[2:]
        x = right - left
        y = bottom - top
        di = (x - y) / (x + y)
        dr = x / y
        return di, dr

    def __calculate_median_x_coordinate_border_end(self, threshold=70):
        """
        Determine median X coordinate of end of border wall in median frame.
        :param median_frame: Image of median frame
        :param threshold: Threshold used for binarization
        :return: Median X coordinate
        """
        img = self.__background.detach().numpy()[0, :, :]
        # img = cv.medianBlur(self.__background.detach().numpy(), 5)
        sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5) ** 2
        x = ((sobel_x / sobel_x.max()) * 255).astype('uint8')
        _, threshold = cv.threshold(x, 70, 255, cv.THRESH_BINARY)
        _, x = (threshold > 0).nonzero()
        median_x = int(np.median(x, 0))
        return median_x

    def has_arrived(self):
        return self.__arrived

    def __set_current_speed(self, movement):
        self.__speed = movement

    def get_current_speed(self):
        return self.__speed

    def get_video_type(self):
        return self.__video_type

    def get_completion_id(self):
        return self.__completion_id

    def get_frame_crops(self):
        return self.__frame_crop_collection
