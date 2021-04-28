import gc
import os
import sys
import imageio
import torch
import numpy as np
import cv2 as cv
from config import PREFIX
from Models import NetTracking, NetExperiment, NetExperimentDAN
from methods import to_3d
from skimage.feature import peak_local_max
from Cell import Cell
from methods import blur

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
    return coordinates, img2


def get_closest_coordinate(prediction: tuple, next_frame_coordinate_list: np.ndarray):
    distances = np.linalg.norm(prediction - next_frame_coordinate_list, axis=1)
    sorted_distance_coordinate = [[distances[i], next_frame_coordinate_list[i]] for i in range(len(distances))]
    sorted_distance_coordinate.sort(key=lambda x: x[0])
    return [i[0] for i in sorted_distance_coordinate], [i[1] for i in
                                                        sorted_distance_coordinate]  # distances, coordinates


def create_new_cells(cell_list: list, coordinates: np.ndarray, video_type: str, background):
    for yx in coordinates:
        if yx[1] <= CELL_INITIALIZATION_THRESHOLD:
            cell = Cell(coordinates=yx, video_type=video_type, median_background=background)
            cell_list.append(cell)


def update_cell_list(cell_list: list, org_coordinates: np.ndarray, frame_width: int, frame_number: int):
    for cell in cell_list:
        if cell.get_prediction()[1] < (frame_width - CELL_JOURNEY_COMPLETION_THRESHOLD):
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


def get_file_extension(path: str) -> tuple:
    filename, ext = os.path.splitext(path)
    filename = filename.split("/")[-1:][0]
    return filename, ext


track_images = dict()


def process_frame(video_path, median_path, frame_num_start, frame_num_end):
    frame_num = 0
    cap = cv.VideoCapture(video_path)
    name, ext = get_file_extension(video_path)
    video_type = name
    median_background = torch.tensor(cv.imread(median_path, 0)).unsqueeze(0) / 255
    cells = []
    track_images[video_type] = [[], []]
    retval, frame = cap.read()
    model = NetExperiment()
    model.load_state_dict(torch.load(model_file))
    model.eval()
    while frame_num < frame_num_start:
        frame_num += 1
        retval, frame = cap.read()
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
        create_new_cells(cell_list=cells, coordinates=gaussian_coordinates, video_type=video_type,
                         background=median_background)
        update_cell_list(cell_list=cells, org_coordinates=gaussian_coordinates,
                         frame_width=width, frame_number=frame_num)
        [cell.make_journey_collage(path_prefix=PREFIX, DAN=True) for cell in cells if cell.has_arrived()]
        cells = [cell for cell in cells if cell.is_alive()]
        # finished_cells += [cell for cell in cells if cell.has_arrived()]
        [cell.extract_segmentation(segmentation=segmentation, frame=frame) for cell in cells]
        [cell.extract_segmentation_DAN(segmentation=segmentation, frame=frame) for cell in cells]
        [cell.draw_personal_prediction(frame=frame) for cell in cells]
        [cell.draw_prediction(frame=frame) for cell in cells]
        write_current_images(track_frame=frame, gaussian_frame=gaussian_frame, frame_num=frame_num,
                             path=PREFIX + video_type + "/",
                             video_type=video_type, save_frames=True)
        frame_num += 1
        del model_input, frame, segmentation, gaussian, gaussian_frame, gaussian_coordinates, width

        retval, frame = cap.read()
    gc.collect()


if __name__ == '__main__':
    video_path = sys.argv[1]
    median_path = sys.argv[2]
    start_index = int(sys.argv[3])
    end_index = int(sys.argv[4])

    process_frame(video_path, median_path, start_index, end_index)
