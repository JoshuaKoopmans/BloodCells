import glob
import os.path
import random
import numpy as np
import torch
import cv2 as cv
import shutil


class Cell:
    def __init__(self, coordinates: np.ndarray, video_type: str):
        self.__coordinates = []
        self.__coordinates.append(self.__check_coordinates(coordinates, True))
        self.__id = random.random() * 10000
        self.__video_type = video_type
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
        self.__color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__segmentation_crop_collection = []
        self.__frame_crop_collection = []
        self.__completion_id = None

    def extract_segmentation(self, segmentation: np.ndarray, frame: np.ndarray):
        frame = frame.copy()
        segmentation = segmentation.copy()
        offset = 25
        desired_shape = (offset * 2, offset * 2)
        current_coordinate = self.get_current_coordinate()
        y = current_coordinate[0]
        x = current_coordinate[1]
        segmentation_crop = segmentation[0, y - offset:y + offset, x - offset:x + offset]
        frame_crop = frame[y - offset:y + offset, x - offset:x + offset][:, :, 0]
        if segmentation_crop.shape == desired_shape and frame_crop.shape == desired_shape:
            self.__segmentation_crop_collection.append(torch.tensor(segmentation_crop))
            self.__frame_crop_collection.append(torch.tensor(frame_crop))

    def make_journey_collage(self, path_prefix=""):
        path = "{}{}/".format(path_prefix, self.get_video_type())

        if not os.path.isdir(path):
            os.mkdir(path)

        frame_crops = torch.cat([item for item in self.__frame_crop_collection], 1)
        segmentation_crops = torch.cat([item for item in self.__segmentation_crop_collection], 1)
        combined_image = torch.cat((frame_crops, segmentation_crops), 0)
        cv.imwrite("{}cell_journey_{}.png".format(path, self.get_completion_id()), combined_image.detach().numpy())
        del self.__segmentation_crop_collection
        del self.__frame_crop_collection
        self.kill()

    def __predict_next_coordinates(self):
        if len(self.get_coordinate_list()) > 1:
            previous_coordinates = self.get_coordinate_list()[-2]
            current_coordinates = self.get_coordinate_list()[-1]
            dy = current_coordinates[0] - previous_coordinates[0]
            dx = current_coordinates[1] - previous_coordinates[1]
            self.__set_current_speed(dx)
            prediction = (current_coordinates[0] + dy, current_coordinates[1] + dx)
            return np.array(prediction)

    def update(self, coordinates):
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

    def compare_coordinate(self, cell2):
        assert isinstance(cell2, Cell)
        distance = np.linalg.norm(self.get_current_coordinate() - cell2.get_current_coordinate(), axis=0)

        if self.get_id() != cell2.get_id():
            if int(distance) < 1:
                self.kill()
                cell2.kill()

    def draw_path(self):
        img = np.zeros((480, 640))
        for i in range(len(self.get_coordinate_list()) - 1):
            img = cv.line(img, pt1=tuple(self.get_coordinate_list()[i][::-1]),
                          pt2=tuple(self.get_coordinate_list()[i + 1][::-1]), color=255)

        cv.imwrite("track_{}.png".format(self.get_id()), (1 - img) * 255)

    def __check_coordinates(self, coordinates, initial=False):
        coordinates = np.array(coordinates)

        assert type(coordinates) is np.ndarray
        if len(coordinates) != 2:
            raise Exception("Coordinates not in correct format.")
        if initial is False:
            if coordinates[1] <= self.get_coordinate_list()[-1][1]:
                return None
        return coordinates

    def draw_personal_prediction(self, frame):
        current_coordinate = tuple(self.get_coordinate_list()[-1][::-1])
        predicted_coordinate = tuple(self.get_prediction()[::-1])
        radius = 30

        frame = frame.copy()
        frame = cv.circle(img=frame, center=current_coordinate, radius=radius, color=self.__color,
                          thickness=2)
        frame = cv.circle(img=frame, center=predicted_coordinate, radius=radius - 15, color=self.__color,
                          thickness=1)
        # frame = cv.arrowedLine(img=frame, pt1=tuple([current_coordinate[0] + radius, current_coordinate[1]]),
        #                        pt2=tuple([predicted_coordinate[0] - radius-15, predicted_coordinate[1]]),
        #                        color=self.__color)
        # cv.imwrite("pred_{}.png".format(str(self.get_id())), frame)
        self.__prediction_images.append(frame)

    def draw_prediction(self, frame):
        current_coordinate = tuple(self.get_coordinate_list()[-1][::-1])
        predicted_coordinate = tuple(self.get_prediction()[::-1])
        radius = 30

        frame = cv.circle(img=frame, center=current_coordinate, radius=radius, color=self.__color,
                          thickness=2)
        frame = cv.circle(img=frame, center=predicted_coordinate, radius=self.__speed, color=self.__color,
                          thickness=1)
        # frame = cv.arrowedLine(img=frame, pt1=tuple([current_coordinate[0] + radius, current_coordinate[1]]),
        #                        pt2=tuple([predicted_coordinate[0] - radius-15, predicted_coordinate[1]]),
        #                        color=self.__color)

    def write_prediction_images(self):
        for i in range(len(self.__prediction_images)):
            cv.imwrite("prediction_{}_{}.png".format(str(self.get_id())[:6], str(i)), self.__prediction_images[i])

    def is_alive(self):
        return self.__alive

    def kill(self):
        self.__alive = False

    def arrived(self, frame_num: int):
        self.__arrived = True
        prediction = [str(yx) for yx in self.get_prediction()]
        self.__completion_id = "{}_{}_{}".format(self.get_video_type(), str(frame_num),
                                                 ";".join(prediction))

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
