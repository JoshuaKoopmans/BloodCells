import gc
import glob
import os
import sys

import imageio
import torch
import numpy as np
import cv2 as cv
from scripts.config import PREFIX
from scripts.Models import NetTracking, NetExperiment
from scripts.methods import to_3d
from skimage.feature import peak_local_max
from Cell import Cell
from scripts.methods import blur

CELL_INITIALIZATION_THRESHOLD = 70
CELL_JOURNEY_COMPLETION_THRESHOLD = 40

medians = {"K1_001_20201105": PREFIX + "median_real_K1.png", "002_2.5kfps": PREFIX + "median_real_002.png",
           "NF135_002_20201105": PREFIX + "median_real_NF.png"}
model_file = PREFIX + "model_b5000_r5000_6390.pt"


def get_coordinates(img, img2=None, text="", verbose=False) -> (torch.Tensor, np.ndarray):
    if type(img) is torch.Tensor or type(img2) is torch.Tensor:
        img = img.detach().numpy()
    img2 = img.copy()
    coordinates = peak_local_max(blur(img, (3, 3)), threshold_abs=.3, min_distance=10)  # (7,7), 0.3, 10
    if verbose:
        n_circles_drawn = len(coordinates)
        print(n_circles_drawn)
        print("Circles drawn: {}".format(n_circles_drawn), end="\n")
    img2 *= 255
    img2[:, CELL_INITIALIZATION_THRESHOLD] = 255
    for coord in coordinates:
        img2 = cv.circle(img2, center=(coord[1], coord[0]), radius=20, thickness=2, color=255)
        if coord[1] <= CELL_INITIALIZATION_THRESHOLD:
            img2 = cv.circle(img2, center=(coord[1], coord[0]), radius=30, thickness=1, color=255)
    # cv.imwrite("circles{}.png".format(text), img2 * 255)
    return coordinates, img2


def get_closest_coordinate(prediction: tuple, next_frame_coordinate_list: np.ndarray):
    distances = np.linalg.norm(prediction - next_frame_coordinate_list, axis=1)
    # minimum_distance = np.amin(distances)
    # minimum_distance_index = np.where(distances == minimum_distance)
    # closest_coordinate = next_frame_coordinate_list[minimum_distance_index][0]
    sorted_distance_coordinate = [[distances[i], next_frame_coordinate_list[i]] for i in range(len(distances))]
    sorted_distance_coordinate.sort(key=lambda x: x[0])

    # print(prediction, minimum_distance, tuple(closest_distance), sep="\t")
    # next_frame_coordinate_list = np.delete(next_frame_coordinate_list, minimum_distance_index, axis=0)
    # next_frame_coordinate_list = []

    # return closest_coordinate, minimum_distance
    return [i[0] for i in sorted_distance_coordinate], [i[1] for i in
                                                        sorted_distance_coordinate]  # distances, coordinates


def create_new_cells(cell_list: list, coordinates: np.ndarray, video_type: str):
    for yx in coordinates:
        if yx[1] <= CELL_INITIALIZATION_THRESHOLD:
            cell = Cell(coordinates=yx, video_type=video_type)
            cell_list.append(cell)


def update_cell_list(cell_list: list, org_coordinates: np.ndarray, frame_width: int, frame_number: int):
    for cell in cell_list:
        if cell.get_prediction()[1] < (frame_width - CELL_JOURNEY_COMPLETION_THRESHOLD):
            # closest_coordinate, closest_distance = get_closest_coordinate(cell.get_prediction(), coordinates)
            distances, coordinates = get_closest_coordinate(cell.get_prediction(), org_coordinates)
            closest_distance, closest_coordinate = distances[0], coordinates[0]

            if closest_distance <= max(30, int(cell.get_current_speed() * 2)):
                distances, coordinates = get_closest_coordinate(closest_coordinate, org_coordinates)
                if len(distances) > 1 and distances[1] < 35:
                    cell.kill()
                cell.update(closest_coordinate)
                del distances, coordinates
            else:

                cell.kill()
            # for other_cell in cell_list:
            #     cell.compare_coordinate(other_cell)
        else:
            cell.arrived(frame_num=frame_number)
        # del distances, coordinates, org_coordinates, closest_coordinate, closest_distance, frame_number


def create_tracking_gif(image_list, gaussian_image_list, path: str, video_type: str, save_frames=False):
    with imageio.get_writer('{}{}_tracking.gif'.format(path, video_type), mode='I',
                            duration=0.5) as writer:  # duration=0.2 (=standard)
        for i in range(len(image_list)):
            image = image_list[i].astype("uint8")
            gaussian_image = gaussian_image_list[i].astype("uint8")
            combined_image = torch.cat((torch.tensor(image), torch.tensor(gaussian_image)), 0).detach().numpy()
            writer.append_data(combined_image)
            if save_frames:
                cv.imwrite("{}{}_{}.png".format(path, video_type, str(i)), combined_image)


def write_current_images(track_frame, gaussian_frame, frame_num: int, path: str, video_type: str, save_frames=False):
    image = track_frame.astype("uint8")
    gaussian_image = gaussian_frame.astype("uint8")
    combined_image = torch.cat((torch.tensor(image), torch.tensor(gaussian_image)), 0).detach().numpy()
    if save_frames:
        cv.imwrite("{}{}_{}.png".format(path, video_type, str(frame_num)), combined_image)


def check_if_dir_exists(path: str):
    if not os.path.isdir(path):
        return False
    return True


def get_single_frame(cap: cv.VideoCapture) -> (bool, np.ndarray):
    return cap.read()


track_images = dict()


def process_frame(video_path, frame_num_start, frame_num_end):
    frame_num = 0
    cap = cv.VideoCapture(video_path)
    video_type = video_path.split("/")[-1][:-4]
    median_background = torch.tensor(cv.imread(medians[video_type], 0)).unsqueeze(0) / 255
    cells = []
    finished_cells = []
    track_images[video_type] = [[], []]
    retval, frame = cap.read()
    model = NetExperiment()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    while frame_num < frame_num_start:
        frame_num += 1

    while retval and frame_num < frame_num_end:
        frame = torch.tensor(frame[:, :, 0]).unsqueeze(0) / 255
        model_input = torch.cat((frame, median_background), 0).unsqueeze(0)
        segmentation, gaussian = model(model_input)
        frame = to_3d(frame.squeeze(0).detach().numpy()) * 255
        segmentation = segmentation.squeeze(0).squeeze(0).detach().numpy() * 255
        gaussian = gaussian.squeeze(0).squeeze(0).detach().numpy()
        gaussian_coordinates, gaussian_frame = get_coordinates(gaussian, text=str(frame_num))
        gaussian_frame = to_3d(gaussian_frame)
        width = gaussian.shape[1]
        create_new_cells(cell_list=cells, coordinates=gaussian_coordinates, video_type=video_type)
        update_cell_list(cell_list=cells, org_coordinates=gaussian_coordinates,
                         frame_width=width, frame_number=frame_num)
        [cell.make_journey_collage(path_prefix=PREFIX) for cell in cells if cell.has_arrived()]
        cells = [cell for cell in cells if cell.is_alive()]
        # finished_cells += [cell for cell in cells if cell.has_arrived()]
        [cell.extract_segmentation(segmentation=segmentation, frame=frame) for cell in cells]
        [cell.draw_personal_prediction(frame=frame) for cell in cells]
        [cell.draw_prediction(frame=frame) for cell in cells]

        # track_images[video_type][0].append(frame)
        # track_images[video_type][1].append(gaussian_frame)

        # cells = [cell for cell in cells if not cell.has_arrived()]

        write_current_images(track_frame=frame, gaussian_frame=gaussian_frame, frame_num=frame_num,
                             path=PREFIX + video_type + "/",
                             video_type=video_type, save_frames=True)
        frame_num += 1
        del model_input, frame, segmentation, gaussian, gaussian_frame, gaussian_coordinates, width

        retval, frame = cap.read()
    gc.collect()




#path = PREFIX + "resources/nice_raw_data/"

process_frame(sys.argv[1], sys.argv[2], sys.argv[3])

# with imageio.get_writer('NF135_002_20201105.avi', mode='I') as writer:  # duration=0.2 (=standard)
#     ret, frames = cv.imreadmulti("/mnt/cellstorage/resources/raw_data/NF135_002_20201105.tif", flags=cv.IMREAD_GRAYSCALE)
#     for f in frames:
#         writer.append_data(f)

# image_type_dir = "K1_001_20201105"
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter(image_type_dir+".avi",fourcc,1,(640,480))
# real_files = [cv.imread(PREFIX + "resources/{}/{}".format(image_type_dir, x), 0) for x in
#                sorted(os.listdir(PREFIX + "resources/{}/".format(image_type_dir)), key=lambda x: int(x.split("_")[-1][:-4]))]
#

# cells = []
# finished_cells = []
#
# medians = {"K1_001_20201105": PREFIX + "median_real_K1.png", "002_2.5kfps": PREFIX + "median_real_002.png",
#            "NF135_002_20201105": PREFIX + "median_real_NF.png"}
# model = NetExperiment()
# model.load_state_dict(torch.load("/home/joshua/Desktop/model_b5000_r5000_6390.pt"))
# model.eval()

# for image_type_dir in os.listdir(PREFIX + "resources/")[:-1]:  # [:-1]: everything except raw_data folder
#     real_files = [torch.tensor(cv.imread(PREFIX + "resources/{}/{}".format(image_type_dir, x), 0)).unsqueeze(0) / 255 for x in
#                   sorted(os.listdir(PREFIX + "resources/{}/".format(image_type_dir)), key=lambda x: int(x.split("_")[-1][:-4]))]
#     median = torch.tensor(cv.imread(medians[image_type_dir], 0)).unsqueeze(0) / 255
#
#     images_track_all_cells = []
#     gaussian_track_all_cells = []
#     cells = []
#     temp_finished_cells = []
#     trash_cells = []
#     for idx, frame in enumerate(real_files):
#         model_input = torch.cat((frame, median), 0).unsqueeze(0)
#         segmentation, gaussian = model(model_input)
#         frame = to_3d(frame.squeeze(0).detach().numpy()) * 255
#         segmentation = segmentation.squeeze(0).squeeze(0).detach().numpy() * 255
#         gaussian = gaussian.squeeze(0).squeeze(0).detach().numpy()
#         gaussian_coordinates, gaussian_frame = get_coordinates(gaussian, text=str(idx))
#         gaussian_frame = to_3d(gaussian_frame)
#         width = gaussian.shape[1]
#         create_new_cells(cell_list=cells, coordinates=gaussian_coordinates, video_type=image_type_dir)
#         update_cell_list(cell_list=cells, org_coordinates=gaussian_coordinates,
#                          frame_width=width, frame_number=idx)
#         cells = [cell for cell in cells if cell.is_alive()]
#         temp_finished_cells += [cell for cell in cells if cell.has_arrived()]
#         [cell.extract_segmentation(segmentation=segmentation, frame=frame) for cell in cells]
#         [cell.draw_personal_prediction(frame=frame) for cell in cells]
#         [cell.draw_prediction(frame=frame) for cell in cells]
#
#         images_track_all_cells.append(frame)
#         gaussian_track_all_cells.append(gaussian_frame)
#         cells = [cell for cell in cells if not cell.has_arrived()]
#         del model_input, frame, segmentation, gaussian, gaussian_frame, gaussian_coordinates, width
#     files = glob.glob(PREFIX + image_type_dir + "/*")
#     for f in files:
#         os.remove(f)
#     create_tracking_gif(images_track_all_cells, gaussian_track_all_cells, path=PREFIX + image_type_dir + "/",
#                         video_type=image_type_dir, save_frames=True)
#
#     [cell.make_journey_collage(path_prefix=PREFIX) for cell in temp_finished_cells]
#
#     print('Analysis on video \"{}\" has been completed.'.format(
#         image_type_dir))
#     gc.collect()
