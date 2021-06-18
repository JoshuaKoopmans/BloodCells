import os
import unittest
import numpy as np
from scripts.Cell import Cell
from scripts.follow_me import get_coordinates, create_new_cells, update_cell_list, CELL_JOURNEY_COMPLETION_THRESHOLD
import cv2 as cv


class TestTrackingRules(unittest.TestCase):
    def setUp(self):
        self.test_gaussians = os.listdir("./tests/gaussian_images/")
        self.test_gaussians.sort(key=lambda x: int(x.split("_")[-1][:-4]))
        self.cell_1 = Cell(coordinates=np.array([610, 10]), video_type="test", median_background=None,
                           initial_frame_num=-1)

    def test_coordinates(self):
        coordinates, _ = get_coordinates(cv.imread("./tests/gaussian_images/" + self.test_gaussians[1], 0) / 255)
        self.assertEqual(len(coordinates), 9)

    def test_cell_initialization(self):
        cell_list = []
        coordinates, _ = get_coordinates(cv.imread("./tests/gaussian_images/" + self.test_gaussians[2], 0) / 255)
        create_new_cells(cell_list, coordinates=coordinates, video_type="test", background=None, frame_num=0)
        self.assertEqual(len(cell_list), 1)

    def test_proximity_kill(self):
        cell_list = []

        for n, frame in enumerate(self.test_gaussians[23:33]):
            coordinates, _ = get_coordinates(cv.imread("./tests/gaussian_images/" + frame, 0) / 255)
            create_new_cells(cell_list=cell_list, coordinates=coordinates, video_type="test", background=None, frame_num=n)
            update_cell_list(cell_list=cell_list, org_coordinates=coordinates, frame_width=640, frame_number=n)

        self.assertEqual(len(cell_list), 5)
        self.assertEqual(set([x.is_alive() for x in cell_list]), {False})

    def test_cell_completion(self):
        for n, frame in enumerate(self.test_gaussians[:]):
            coordinates, _ = get_coordinates(cv.imread("./tests/gaussian_images/" + frame, 0) / 255)
        if self.cell_1.get_current_coordinate()[1] < (640 - CELL_JOURNEY_COMPLETION_THRESHOLD):
            self.cell_1.arrived(frame_num=42)
        self.assertEqual(self.cell_1.has_arrived(), True)


if __name__ == '__main__':
    unittest.main()
