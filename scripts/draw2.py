import imageio
import numpy as np
import cv2 as cv
import random
import sys
import torch
import time

from skimage.segmentation import flood_fill
from multiprocessing import Pool, cpu_count
from methods import *

random.seed(0)
randomness = random.randint

if os.environ.get("PREFIX") is None:
    prefix = "/media/joshua/Red Blood Cells/"
else:
    prefix = os.environ.get("PREFIX")

def round_red_blood_cell(image):
    image = np.zeros((100, 100))
    try:
        assert len(image.shape) == 2
    except AssertionError:
        print("ERROR: Expected 2d input, got something else.\n Exiting...")
        sys.exit(1)
    width, height = image.shape[0], image.shape[1]
    seeds = [int(random.random() * 1000000), int(random.random() * 1000000)]

    ############
    center = (50, 50)
    outer_radius = 16
    middle_radius = 14
    inner_radius = 13
    outer_thickness_variation = random.randint(3 - 2, 3 + 1)
    middle_thickness_variation = random.randint(2 - 1, 2 + 1)

    ############
    image = add_circle(image=image, center=center, radius=outer_radius, color=250, thickness=outer_thickness_variation)

    interpol = minmax(genNoise(width, 0.015, seeds[0]), 0, 1)[:, :, 0]
    tmp = interpol * image + (np.ones((width, height)) + 110) * (1 - interpol)

    image[image == 250] = tmp[image == 250]

    image = add_circle(image=image, center=center, radius=middle_radius, color=0, thickness=middle_thickness_variation)
    image = add_circle(image=image, center=center, radius=inner_radius, color=70, thickness=-1)

    ## kern ##
    kern = np.zeros((width, height))
    kern = add_circle(kern, (50 + random.randint(-5, 5), 50 + random.randint(-5, 5)), 5, 255, -1)
    kern = (255 - elastic(kern, 300, 12)) / 255
    if random.random() > 0.9:
        warp_num = 12
    else:
        warp_num = 8
    noise = minmax(genNoise(width, 0.2, seeds[0]), 0, 220)[:, :, 0]
    image[image == 70] = noise[image == 70]
    image *= kern
    image = elastic(image, 200, warp_num, seeds[1])

    ### MASK ###
    mask = np.zeros((width, height))
    mask = add_circle(mask, center, outer_radius, 255, outer_thickness_variation)
    mask = add_circle(mask, center, middle_radius, 255, middle_thickness_variation)
    mask = add_circle(mask, center, inner_radius, 200, -1)
    mask = elastic(mask, 200, warp_num, seeds[1])
    label = mask.copy()
    if random.random() < 0.35:
        mask[mask > 210] *= minmax(genNoise(width, 0.3)[:, :, 0], 0, 1.1)[mask > 210]
    # imageio.imwrite("mask.png", mask.astype("uint8"))
    # imageio.imwrite("label.png", label.astype("uint8"))

    image[mask < 120] = -10
    # image = blur(image, (3, 3))
    image[mask < 120] += minmax(genNoise(width, 10, seeds[0]) ** 4, -15, +8)[:, :, 0][mask < 120]

    ### BACKGROUND ###

    bg, alt = background(width=width, height=height, seed=seeds[0], alternative=True)

    ### EXTRACT CELL w/ MASK  ###
    cell = np.zeros((width, height))
    cell[image > 0] = image[image > 0]

    cell = blur(cell, (3, 3))
    mask = blur(mask, (3, 3))
    mask = mask / 255.
    cell = cell * mask + (bg * (1 - mask))

    # imageio.imwrite("cell.png", cell.astype("uint8"))

    image[image > 255] = 255
    image[image < 0] = 0
    label[label > 0] = 255
    # imageio.imwrite("label.png", label.astype("uint8"))
    if random.random() > 0.5:
        cell = bg
        label = np.zeros((width, height))
    if np.sum(label) != 0:
        label_gauss = generate_instancing_labels(label)
        label_gauss[label_gauss < 0] = 0
        label_gauss[label_gauss > 255] = 255
    else:
        label_gauss = label

    return cell, label, alt, label_gauss


def bullet_red_blood_cell(image):
    image = np.zeros((100, 100))
    try:
        assert len(image.shape) == 2
    except AssertionError:
        print("ERROR: Expected 2d input, got something else.\n Exiting...")
        sys.exit(1)
    width, height = image.shape[0], image.shape[1]
    seeds = [int(random.random() * 1000000), int(random.random() * 1000000)]

    if random.random() > 0.5:
        inner_color = randomness(30, 80)
        outer_color = randomness(180, 250)
    else:
        inner_color = randomness(120, 200)
        outer_color = randomness(30, 80)

    cell, memory_seed = create_bullet_cell(image=image, radius=10, x_offset=-15, color=inner_color, thickness=2,
                                           seed=seeds[0], memory=False)

    cell = create_bullet_cell(image=image, radius=10, x_offset=-15, color=outer_color, thickness=2, memory=True,
                              seed=memory_seed)
    center_x = [x for x in range(0, width) if cell[int(width / 2), x] != 0]
    center = (50, int(sum(center_x) / len(center_x)))

    cell_border = cell.copy()
    label = cell.copy()
    label[cell_border == outer_color] = 255

    # imageio.imwrite("label_b.png", label.astype("uint8"))
    label = flood_fill(label, center, 255)
    if random.random() > 0.3:
        angle = randomness(-30, 30)
        cell = rotate_image(cell, angle)
        cell_border = rotate_image(cell_border, angle)
        label = rotate_image(label, angle)
        # label_gauss = rotate_image(label_gauss, angle)
    cell_border[cell != 0] = 255 - cell_border[cell != 0] + randomness(-30, 40)
    cell_border = dilate(cell_border, (3, 3))

    mask_complete = cell_border.copy()

    mask_complete[cell_border > 0] = 255
    cell = flood_fill(cell, center, inner_color)

    mask_inner = cell.copy()
    mask_inner[cell != inner_color] = 0
    mask_inner[cell == inner_color] = 255
    background_variation = randomness(-15, 30)
    mask_complete[mask_inner == 255] = randomness(100 + background_variation, 110 + background_variation)

    cell_border = blur(cell_border, (9, 9))
    mask_complete = blur(mask_complete, (9, 9))
    cell[cell == 0] = cell_border[cell == 0]
    cell = blur(cell)
    cell = elastic(cell, 400, 17, seed=seeds[1])
    label = elastic(label, 400, 17, seed=seeds[1])

    bg, alt = background(width=width, height=height, seed=seeds[0], alternative=True)

    mask_complete = elastic(mask_complete, 400, 17, seed=seeds[1])
    mask_complete = mask_complete / 255.
    cell = cell * mask_complete + (bg * (1 - mask_complete))
    cell[cell < 0] = 0
    cell[cell > 255] = 255

    if random.random() > 0.8:
        cell = bg
        label = np.zeros((width, height))
    if np.sum(label) != 0:
        label_gauss = generate_instancing_labels(label)
        label_gauss[label_gauss < 0] = 0
        label_gauss[label_gauss > 255] = 255
    else:
        label_gauss = label
    return cell, label, alt, label_gauss


def genImg(func):
    p = Pool(10)
    imgs = p.map(func, range(16))
    cells, labels, background, gauss_labels = [], [], [], []
    for image in imgs:
        cells.append(image[0])
        labels.append(image[1])
        background.append(image[2])
        gauss_labels.append(image[3])

    size = imgs[0][0].shape[0]
    cell = np.zeros((size * 4, size * 4))
    label = np.zeros((size * 4, size * 4))
    bg = np.zeros((size * 4, size * 4))
    gauss = np.zeros((size * 4, size * 4))
    index = set(range(0, 17, 1))

    for x in range(4):
        for y in range(4):
            count = index.pop()
            cell[x * size:(x + 1) * size, y * size:(y + 1) * size] = cells[count]
            label[x * size:(x + 1) * size, y * size:(y + 1) * size] = labels[count]
            bg[x * size:(x + 1) * size, y * size:(y + 1) * size] = background[count]
            gauss[x * size:(x + 1) * size, y * size:(y + 1) * size] = gauss_labels[count]
    imageio.imwrite("cell_grid.png", cell.astype('uint8'))
    imageio.imwrite("label_grid.png", label.astype('uint8'))
    imageio.imwrite("background_grid.png", bg.astype('uint8'))
    imageio.imwrite("gauss_label_grid.png", gauss.astype('uint8'))


def generate_images(func, out_filename: str):
    available_thread_count = cpu_count()
    start = time.time()
    p = Pool(available_thread_count)

    imgs = p.map(func, range(5000))
    data = [torch.cat(((image[0].unsqueeze(0)), image[2].unsqueeze(0)), 0) for image in torch.tensor(imgs)]
    data = torch.cat([i.unsqueeze(0) for i in data], 0)
    torch.save(data, out_filename + "_data.pt")

    label = [(image[1].unsqueeze(0)) for image in torch.tensor(imgs)]
    label = torch.cat([i.unsqueeze(0) for i in label], 0)
    torch.save(label, out_filename + "_labels.pt")

    label_gauss = [(image[3].unsqueeze(0)) for image in torch.tensor(imgs)]
    label_gauss = torch.cat([i.unsqueeze(0) for i in label_gauss], 0)
    torch.save(label_gauss, out_filename + "_labels_gauss.pt")

    end = time.time() - start


#genImg(round_red_blood_cell)


# genImg(bullet_red_blood_cell)

def heat(img):
    new = torch.zeros(img.shape[0], img.shape[1], 3)
    red = torch.tensor([128, 0, 0]).reshape(-1, 1)
    blue = torch.tensor([0, 0, 128]).reshape(-1, 1)
    white = torch.tensor([255, 255, 255]).reshape(-1, 1)
    new[img < 0.5] = ((img[img < .5] * 2) * white + (1 - (img[img < .5] * 2)) * blue).permute(1, 0)
    new[img > 0.5] = (((img[img > .5] - .5) * 2) * red + (1 - ((img[img > .5] - .5) * 2)) * white).permute(1, 0)
    imageio.imwrite('heatmap.png', new.numpy().astype('uint8'))


#heat(torch.tensor(cv.imread('evalutate.png')[:, :, 0]) / 255.)
#prefix = "/media/joshua/Red Blood Cells/"
generate_images(round_red_blood_cell, prefix + "round_cells")
generate_images(bullet_red_blood_cell, prefix + "bullet_cells")
