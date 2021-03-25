import os
import sys
import random
import numpy as np
import torch.nn as F
import torch
import cv2 as cv


def open_save_video_files(raw_path="../resources/raw_data/", out_path="../resources/", allowed_ext=None, verbose=False):
    if allowed_ext is None:
        allowed_ext = ["tiff", "tif", "avi", "mp4"]
    try:
        directory_content = os.listdir(raw_path)
    except FileNotFoundError:
        print("Given path does not exist. -->", sys.exc_info()[1])
        sys.exit(1)
    for file in directory_content:
        KNN = cv.createBackgroundSubtractorKNN()
        MOG2 = cv.createBackgroundSubtractorMOG2()
        video_path = os.path.join(raw_path, file)
        if os.path.isfile(video_path):
            filename, ext = get_file_extension(video_path)
            if extension_is_allowed(ext, allowed_ext):
                if verbose:
                    print("Extracting frames from {}...".format(video_path))
                if is_tiff(ext):
                    frames = read_tiff(video_path)
                else:
                    frames = read_video(video_path)
                directory_name = os.path.join(out_path, filename)
                if not os.path.exists(directory_name):
                    os.makedirs(directory_name)

                i = 0
                for frame in frames:
                    MOG2bg, KNNbg = update_foreground_mask(frame, MOG2, KNN)
                    output_path = "{}/{}_{}.png".format(directory_name, filename, str(i))
                    cv.imwrite(output_path, frame)
                    cv.imwrite("{}/{}/MOG2_bg_{}_{}.png".format("MOG2", filename, filename, str(i)), MOG2bg)
                    cv.imwrite("{}/{}/KNN_bg_{}_{}.png".format("KNN", filename, filename, str(i)), KNNbg)
                    i += 1
                if verbose:
                    print("Extracted {} frames.\nDone.".format(str(i + 1)))

            else:
                raise Exception("Input file {} not supported".format(file))


def get_file_extension(path: str) -> tuple:
    filename, ext = os.path.splitext(path)
    filename = filename.split("/")[-1:][0]
    return filename, ext


def extension_is_allowed(ext: str, allowed_ext: list) -> bool:
    if ext[1:].lower() in allowed_ext:
        return True
    else:
        return False


def is_tiff(ext: str) -> bool:
    return ext[1:] in ["tif", "tiff"]


def read_tiff(image_path: str):
    ret, frames = cv.imreadmulti(image_path)
    return frames


def read_video(video_path: str):
    frames = []
    cap = cv.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    cv.destroyAllWindows()
    return frames


def update_foreground_mask(frame, fgmaskMOG2obj, fgmaskKNNobj):
    fgmaskKNN = fgmaskKNNobj.apply(frame)
    fgmaskMOG2 = fgmaskMOG2obj.apply(frame)

    return fgmaskKNN, fgmaskMOG2
    # cv.imwrite("{}.jpg".format(names[i]), bg)
    # i += 1


# open_save_video_files(verbose=True)

def remove_background(path: str):
    frames = []
    mean_frame = []
    median_frame = []

    try:
        directory_content = os.listdir(path)
    except FileNotFoundError:
        print("Given path does not exist. -->", sys.exc_info()[1])
        sys.exit(1)
    for i in range(1000):
        frame = random.choice(directory_content)
        frame = os.path.join(path, frame)
        frames.append(cv.imread(frame)[:, :, 0])

    frames = torch.tensor(np.array(frames)).float()

    test_img = torch.tensor(cv.imread("../resources/002_2.5kfps/002_2.5kfps_72.png")[:, :, 0]).float()

    mean = torch.mean(frames, 0)
    median = torch.median(frames, 0)[0]

    cv.imwrite("MEAN.png", mean.numpy())
    cv.imwrite("MEDIAN.png", median.numpy())
    cv.imwrite("mean.png", ((mean - test_img) ** 2).numpy())
    cv.imwrite("median.png", ((median - test_img) ** 2).numpy())

    m = F.MaxPool2d(3)
    pool_median = (m(torch.tensor(test_img).unsqueeze(0).unsqueeze(0)) - m(torch.tensor(median).unsqueeze(0).unsqueeze(0)))**2
    pool_mean = (m(torch.tensor(test_img).unsqueeze(0).unsqueeze(0)) - m(torch.tensor(mean).unsqueeze(0).unsqueeze(0)))**2
    cv.imwrite("MaxPool-mean.png", pool_mean.numpy()[0][0])
    cv.imwrite("MaxPool-median.png", pool_median.numpy()[0][0])
    cv.waitKey(9999999)
    # print(np.median(frames, -1))
    # mean_frame = np.array([np.matrix(x) / len(frames) for x in frames])
    # print(mean_frame)

open_save_video_files(verbose=True)
#remove_background("../resources/002_2.5kfps")
"""
mean of (100, 1000 random) frames
median of (100, 1000 random) frames
abs(frame - mean|median) or  (frame - mean|median)**2
abs(max_pool(frame) - max_pool(median))

200 220  -> -20
200 180  -> +20
"""
