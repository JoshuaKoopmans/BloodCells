import gc
import os
import sys
import imageio
import torch
import numpy as np
import cv2 as cv
from scripts.config import PREFIX
from scripts.Models import NetExperiment
from scripts.methods import to_3d, blur
from skimage.feature import peak_local_max
from scripts.Cell import Cell

CELL_INITIALIZATION_THRESHOLD = 70
CELL_JOURNEY_COMPLETION_THRESHOLD = 40

model_file = PREFIX + "model_b5000_r5000_6390.pt"


def get_coordinates(img, verbose=False) -> (torch.Tensor, np.ndarray):
    """
    Get coordinates of instancing labels predicted by neural network.
    Add circles to identify instancing predictions above a 30% probability
    :param img: Instancing prediction frame
    :param verbose: Print amounts of coordinates identified
    :return: Image with circles for identified peaks
    """
    if type(img) is torch.Tensor:
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
    """
    Given all coordinates for a given frame, use the euclidean distance to determine the closest to a specific coordinate
    :param prediction: Coordinate prediction of a specific cell
    :param next_frame_coordinate_list: List of all actual cell coordinates in the next frame
    :return: List of coordinates with their distance to the prediction
    """
    distances = np.linalg.norm(prediction - next_frame_coordinate_list, axis=1)
    sorted_distance_coordinate = [[distances[i], next_frame_coordinate_list[i]] for i in range(len(distances))]
    sorted_distance_coordinate.sort(key=lambda x: x[0])
    return [i[0] for i in sorted_distance_coordinate], [i[1] for i in
                                                        sorted_distance_coordinate]  # distances, coordinates


def create_new_cells(cell_list: list, coordinates: np.ndarray, video_type: str, background, frame_num):
    """
    Add new cells to cell list
    :param cell_list: List of active cell objects
    :param coordinates: List of cell coordinates for next frame
    :param video_type: Name of type of video being analyzed
    :param background: Median frame of video
    :param frame_num: Frame number
    """
    for yx in coordinates:
        if yx[1] <= CELL_INITIALIZATION_THRESHOLD:
            cell = Cell(coordinates=yx, video_type=video_type, median_background=background, initial_frame_num=frame_num)
            cell_list.append(cell)


def update_cell_list(cell_list: list, org_coordinates: np.ndarray, frame_width: int, frame_number: int):
    """
    For all active cell objects, update cell or kill cell
    :param cell_list: List of active cell objects
    :param org_coordinates: List of cell coordinates for next frame
    :param frame_width: Frame width
    :param frame_number: Frame number
    """
    for cell in cell_list:
        if cell.get_prediction()[1] < (frame_width - CELL_JOURNEY_COMPLETION_THRESHOLD):
            distances, coordinates = get_closest_coordinate(cell.get_prediction(), org_coordinates)
            try:
                closest_distance, closest_coordinate = distances[0], coordinates[0]
            except IndexError:
                cell.kill()
                continue
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


def create_tracking_gif(image_list, gaussian_image_list, path: str, video_type: str, save_frames=False):
    """
    Using the frames of cells with circles around them resulting from tracking, create a gif.
    :param image_list: List of real frames with circles
    :param gaussian_image_list: List of instancing labels of the real frames
    :param path: Path where to save gif
    :param video_type: Name of type of video
    :param save_frames: Whether to save individual frames
    """
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
    """
    Concat real frame with circles due to tracking with the instancing label and save.
    :param track_frame: Real frame with circles
    :param gaussian_frame: Instancing label of the real frame
    :param frame_num: Frame number
    :param path: Path where to save gif
    :param video_type: Name of type of video
    :param save_frames: Whether to save individual frames
    """
    image = track_frame.astype("uint8")
    gaussian_image = gaussian_frame.astype("uint8")
    combined_image = torch.cat((torch.tensor(image), torch.tensor(gaussian_image)), 0).detach().numpy()
    if save_frames:
        cv.imwrite("{}{}_{}.png".format(path, video_type, str(frame_num)), combined_image)


def check_if_dir_exists(path: str):
    """
    Check if a directory exists
    :param path: Path to directory
    :return: boolean
    """
    if not os.path.isdir(path):
        return False
    return True


def get_single_frame(cap: cv.VideoCapture) -> (bool, np.ndarray):
    """
    Returns a single frame from a opened video object
    :param cap: video object
    :return: numpy array of image
    """
    return cap.read()


def get_file_extension(path: str) -> tuple:
    """
    Given a path, retrieve file extension
    :param path: Path
    :return: Filename and extension
    """
    filename, ext = os.path.splitext(path)
    filename = filename.split("/")[-1:][0]
    return filename, ext


track_images = dict()


def process_frame(video_path, median_path, frame_num_start, frame_num_end):
    """
    Track cells in a chunk of frames
    :param video_path: Path to video
    :param median_path: Path to corresponding video median frame
    :param frame_num_start: Start frame
    :param frame_num_end: End frame
    """
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
    while int(frame_num) < int(frame_num_start):
        frame_num += 1
        retval, frame = cap.read()
    while retval and int(frame_num) < int(frame_num_end):
        frame = torch.tensor(frame[:, :, 0]).unsqueeze(0) / 255
        model_input = torch.cat((frame, median_background), 0).unsqueeze(0)
        segmentation, gaussian = model(model_input)
        frame = to_3d(frame.squeeze(0).detach().numpy()) * 255
        segmentation = segmentation.squeeze(0).squeeze(0).detach().numpy() * 255
        gaussian = gaussian.squeeze(0).squeeze(0).detach().numpy()
        gaussian_coordinates, gaussian_frame = get_coordinates(gaussian)
        gaussian_frame = to_3d(gaussian_frame)
        width = gaussian.shape[1]
        create_new_cells(cell_list=cells, coordinates=gaussian_coordinates, video_type=video_type,
                         background=median_background, frame_num=frame_num)
        update_cell_list(cell_list=cells, org_coordinates=gaussian_coordinates,
                         frame_width=width, frame_number=frame_num)
        [cell.make_journey_collage(path_prefix=PREFIX) for cell in cells if cell.has_arrived()]
        cells = [cell for cell in cells if cell.is_alive()]
        # finished_cells += [cell for cell in cells if cell.has_arrived()]
        [cell.extract_segmentation(segmentation=segmentation, frame=frame) for cell in cells]
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
