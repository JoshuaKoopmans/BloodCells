import os.path
import random
import numpy as np
import torch
import cv2 as cv


class Cell:
    def __init__(self, coordinates: np.ndarray, video_type: str, median_background: torch.tensor, initial_frame_num: int):
        self.__coordinates = []
        self.__coordinates.append(self.__check_coordinates(coordinates, True))
        self.__id = random.random() * 10000
        self.__video_type = video_type
        self.__background = median_background
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
        random.seed(coordinates.sum()+initial_frame_num)
        self.__color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.__segmentation_crop_collection = []
        self.__frame_crop_collection = []
        self.__clean_background_crop_collection = []
        self.__DAN_segmentation_crop_collection = []
        self.__DAN_frame_crop_collection = []
        self.__DAN_clean_background_crop_collection = []
        self.__completion_id = None

    def extract_segmentation(self, segmentation: np.ndarray, frame: np.ndarray):
        frame = frame.copy()
        segmentation = segmentation.copy()
        offset = 30
        desired_shape = (offset * 2, offset * 2)
        current_coordinate = self.get_current_coordinate()
        y = current_coordinate[0]
        x = current_coordinate[1]
        segmentation_crop = segmentation[0, y - offset:y + offset, x - offset:x + offset]
        frame_crop = frame[y - offset:y + offset, x - offset:x + offset][:, :, 0]
        clean_background_crop = self.__background.detach().numpy()[0, y - offset:y + offset, x - offset:x + offset]
        if segmentation_crop.shape == desired_shape and frame_crop.shape == desired_shape and clean_background_crop.shape == desired_shape:
            self.__segmentation_crop_collection.append(torch.tensor(segmentation_crop))
            self.__frame_crop_collection.append(torch.tensor(frame_crop))
            self.__clean_background_crop_collection.append(torch.tensor(clean_background_crop))
    def extract_segmentation_DAN(self, segmentation: np.ndarray, frame: np.ndarray):
        frame = frame.copy()
        segmentation = segmentation.copy()
        offset = 50
        desired_shape = (offset * 2, offset * 2)
        current_coordinate = self.get_current_coordinate()
        y = current_coordinate[0]
        x = current_coordinate[1]
        segmentation_crop = segmentation[0, y - offset:y + offset, x - offset:x + offset]
        frame_crop = frame[y - offset:y + offset, x - offset:x + offset][:, :, 0]
        clean_background_crop = self.__background.detach().numpy()[0, y - offset:y + offset, x - offset:x + offset]
        if segmentation_crop.shape == desired_shape and frame_crop.shape == desired_shape and clean_background_crop.shape == desired_shape:
            self.__DAN_segmentation_crop_collection.append(torch.tensor(segmentation_crop))
            self.__DAN_frame_crop_collection.append(torch.tensor(frame_crop))
            self.__DAN_clean_background_crop_collection.append(torch.tensor(clean_background_crop))

    def make_journey_collage(self, path_prefix="", DAN=False):
        path = "{}{}/".format(path_prefix, self.get_video_type())

        if not os.path.isdir(path):
            os.mkdir(path)
        if DAN:
            DAN_frames = torch.cat([item for item in self.__DAN_frame_crop_collection], 1)
            DAN_background = torch.cat([item*255 for item in self.__DAN_clean_background_crop_collection], 1)
            DAN_segmentation = torch.cat([item for item in self.__DAN_segmentation_crop_collection], 1)
            combined_image = torch.cat((DAN_frames, DAN_background, DAN_segmentation), 0)
            cv.imwrite("{}cell_journey_DAN_{}.png".format(path, self.get_completion_id()),
                       combined_image.detach().numpy())

        frame_crops = torch.cat([item for item in self.__frame_crop_collection], 1)
        segmentation_crops = torch.cat([item for item in self.__segmentation_crop_collection], 1)
        background_crop = torch.cat([item for item in self.__clean_background_crop_collection], 1)
        combined_image = torch.cat((frame_crops, background_crop, segmentation_crops), 0)
        cv.imwrite("{}cell_journey_{}.png".format(path, self.get_completion_id()), combined_image.detach().numpy())
        del self.__segmentation_crop_collection, self.__frame_crop_collection, self.__clean_background_crop_collection
        del self.__DAN_clean_background_crop_collection, self.__DAN_segmentation_crop_collection, self.__DAN_frame_crop_collection
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
