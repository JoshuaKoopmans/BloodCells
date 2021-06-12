import imageio
import numpy as np
import cv2 as cv
import random
import sys
import torch
import os
from skimage.segmentation import flood_fill
from multiprocessing import Pool, cpu_count
from methods import minmax, genNoise, blur, elastic, rotate_image, create_bullet_cell, create_shadow_bullet_cells, \
    create_straight_D_cells, generate_instancing_labels, add_circle, background, dilate

"""
This module contains the functions to make the two different types of synthetic data: circle cells and bullet-like cells.
For both cell types, segmentation and instancing labels and background are created.
"""

random.seed(0)
randomness = random.randint

if os.environ.get("PREFIX") is None:
    prefix = "/mnt/cellstorage/"
else:
    prefix = os.environ.get("PREFIX")


def round_red_blood_cell():
    """
    Function to create synthetic round red blood cells
    by manipulating a combination of basic circles with transformations and effects
    (e.g., warping blurring, noise textures).

    Returns: Cell, background, segmentation label, instancing label
    """
    image = np.zeros((100, 100))
    try:
        assert len(image.shape) == 2
    except AssertionError:
        print("ERROR: Expected 2d input, got something else.\n Exiting...")
        sys.exit(1)

    ######################################################
    # Define parameters to be used later on              #
    ######################################################
    width, height = image.shape[0], image.shape[1]
    seeds = [int(random.random() * 1000000), int(random.random() * 1000000)]
    center = (50, 50)
    outer_radius = 16
    middle_radius = 14
    inner_radius = 13
    outer_thickness_variation = random.randint(3 - 2, 3 + 1)
    middle_thickness_variation = random.randint(2 - 1, 2 + 1)

    ######################################################
    # Draw the initial circles and add a color gradient  #
    # to the outline with the noise function.            #
    ######################################################
    image = add_circle(image=image, center=center, radius=outer_radius, color=250, thickness=outer_thickness_variation)
    interpol = minmax(genNoise(width, 0.015, seeds[0]), 0, 1)[:, :, 0]
    tmp = interpol * image + (np.ones((width, height)) + 110) * (1 - interpol)
    image[image == 250] = tmp[image == 250]
    image = add_circle(image=image, center=center, radius=middle_radius, color=0, thickness=middle_thickness_variation)
    image = add_circle(image=image, center=center, radius=inner_radius, color=70, thickness=-1)

    ######################################################
    # Add malaria-like parasites and a noise texture     #
    # to the inside of the circle and warp the parasite. #
    ######################################################
    ## BLACK CIRCLE IN NUCLEUS ##
    kern = np.zeros((width, height))
    kern = add_circle(kern, (50 + random.randint(-5, 5), 50 + random.randint(-5, 5)), 5, 255, -1)
    kern = (255 - elastic(kern, 300, 12)) / 255
    if random.random() > 0.9:
        warp_num = 12
    else:
        warp_num = 8
    ### NUCLEUS NOISE ###
    if random.random() > 0.5:
        noise_color_min = 50
        noise_color_max = 100
    else:
        noise_color_min = 0
        noise_color_max = 220
    noise = minmax(genNoise(width, 0.2, seeds[0]), noise_color_min, noise_color_max)[:, :, 0]
    image[image == 70] = noise[image == 70]

    image *= kern
    image = elastic(image, 200, warp_num, seeds[1])

    #########################################
    # Create mask for the cell, warp cell   #
    #########################################
    mask = np.zeros((width, height))
    mask = add_circle(mask, center, outer_radius, 255, outer_thickness_variation)
    mask = add_circle(mask, center, middle_radius, 255, middle_thickness_variation)
    mask = add_circle(mask, center, inner_radius, 200, -1)
    mask = elastic(mask, 200, warp_num, seeds[1])
    label = mask.copy()
    if random.random() < 0.35:
        mask[mask > 210] *= minmax(genNoise(width, 0.3)[:, :, 0], 0, 1.1)[mask > 210]

    image[mask < 120] = -10
    image[mask < 120] += minmax(genNoise(width, 10, seeds[0]) ** 4, -15, +8)[:, :, 0][mask < 120]

    #########################################
    # Generate background for use with cell #
    #########################################
    bg, alt = background(width=width, height=height, seed=seeds[0], alternative=True)

    ######################################################
    # Blur cell and label & combine cell with background #
    ######################################################
    cell = np.zeros((width, height))
    cell[image > 0] = image[image > 0]

    cell = blur(cell, (3, 3))
    mask = blur(mask, (3, 3))
    mask = mask / 255.
    cell = cell * mask + (bg * (1 - mask))

    image[image > 255] = 255
    image[image < 0] = 0
    label[label > 0] = 255

    #########################################
    # Sometimes give background as cell     #
    #########################################
    if random.random() > 0.8:
        cell = bg
        label = np.zeros((width, height))

    #########################################
    # Generate segmentation labels          #
    #########################################
    if np.sum(label) != 0:
        label_gauss = generate_instancing_labels(label)
        label_gauss[label_gauss < 0] = 0
        label_gauss[label_gauss > 255] = 255
    else:
        label_gauss = label

    return cell, label, alt, label_gauss


def bullet_red_blood_cell():
    """
    Function to create synthetic bullet-like red blood cells
    by manipulating a combination of basic half-circles connected by lines with transformations and effects
    (e.g., warping blurring, noise textures).

    Returns: Cell, background, segmentation label, instancing label
    """
    image = np.zeros((100, 100))
    try:
        assert len(image.shape) == 2
    except AssertionError:
        print("ERROR: Expected 2d input, got something else.\n Exiting...")
        sys.exit(1)

    ######################################################
    # Define parameters to be used later on              #
    ######################################################
    width, height = image.shape[0], image.shape[1]
    seeds = [int(random.random() * 1000000), int(random.random() * 1000000)]

    if random.random() > 0.5:
        inner_color = randomness(30, 80)
        outer_color = randomness(180, 250)
    else:
        inner_color = randomness(120, 200)
        outer_color = randomness(30, 80)

    ###############################################################
    # Generate initial bullet-like shapes                         #
    # with various variation (e.g., D-shaped, cells with shadows) #
    # based on random chances                                     #
    ###############################################################
    if random.random() > 0.6:
        if random.random() > 0.7:
            cell, memory_seed = create_bullet_cell(image=image, radius=10, x_offset=-15, color=outer_color, thickness=2,
                                                   seed=seeds[0], memory=False)
            cell = create_bullet_cell(image=image, radius=10, x_offset=-15, color=outer_color, thickness=2,
                                      seed=memory_seed, memory=True)
        else:
            cell, memory_seed = create_straight_D_cells(image=image, radius=10, x_offset=-15, color=inner_color,
                                                        thickness=2,
                                                        seed=seeds[0], memory=False)
            cell = create_straight_D_cells(image=image, radius=10, x_offset=-15, color=outer_color, thickness=2,
                                           memory=True,
                                           seed=memory_seed)
    else:
        cell, memory_seed = create_shadow_bullet_cells(image=image, radius=10, x_offset=-15, color=inner_color,
                                                       thickness=2,
                                                       seed=seeds[0], memory=False)

        shadow_cell = create_shadow_bullet_cells(image=image, radius=10, x_offset=-15, color=outer_color,
                                                 thickness=2, memory=True,
                                                 seed=memory_seed)

    ###########################################################
    # Center of cell is determined to make segmentation label #
    ###########################################################
    center_x = [x for x in range(0, width) if cell[int(width / 2), x] != 0]
    center = (50, int(sum(center_x) / len(center_x)))
    label = cell.copy()
    try:
        cell_border = shadow_cell.copy()
    except Exception:
        cell_border = cell.copy()

    label[label == outer_color] = 255
    label = flood_fill(label, center, 255)

    #################################################
    # Cells and labels are rotated by random chance #
    #################################################
    if random.random() > 0.3:
        angle = randomness(-30, 30)
        cell = rotate_image(cell, angle)
        cell_border = rotate_image(cell_border, angle)
        label = rotate_image(label, angle)
    cell_border[cell != 0] = 255 - cell_border[cell != 0] + randomness(-30, 40)
    cell_border = dilate(cell_border, (3, 3))

    mask_complete = label.copy()

    mask_complete[cell_border > 0] = 255
    cell = flood_fill(cell, center, inner_color)

    mask_inner = cell.copy()
    mask_inner[cell != inner_color] = 0
    mask_inner[cell == inner_color] = 255

    ############################################################
    # Inside of cell is filled with a color or a noise texture #
    ############################################################
    if random.random() > 0.5:
        background_variation = randomness(-15, 30)
        mask_complete[mask_inner == 255] = randomness(100 + background_variation, 110 + background_variation)
    else:
        ### NUCLEUS NOISE ###
        if random.random() > 0.5:
            noise_color_min = 50
            noise_color_max = 100
        else:
            noise_color_min = 0
            noise_color_max = 220
        noise = minmax(genNoise(width, 0.2, seeds[0]), noise_color_min, noise_color_max)[:, :, 0]
        cell[mask_inner == 255] = noise[mask_inner == 255]
        ######
        # cell[mask_inner == 255] = kern[mask_inner == 255]
        # cell = cell* kern + mask_complete* (1-kern)
        # cell[mask_inner == 255] *= kern[mask_inner == 255]

    ############################################################
    # Blur, warp cell and label & combine cell with background #
    ############################################################
    cell_border = blur(cell_border, (9, 9))
    mask_complete = blur(mask_complete, (9, 9))
    cell[cell == 0] = cell_border[cell == 0]
    cell = blur(cell)
    cell = elastic(cell, 400, 17, seed=seeds[1])
    label = elastic(label, 400, 17, seed=seeds[1])

    #########################################
    # Generate background for use with cell #
    #########################################
    bg, alt = background(width=width, height=height, seed=seeds[0], alternative=True)

    mask_complete = elastic(mask_complete, 400, 17, seed=seeds[1])
    mask_complete = mask_complete / 255.
    cell = cell * mask_complete + (bg * (1 - mask_complete))

    # Random horizontal or vertical line with varying degrees of invisibility
    mask_line = np.zeros((height, width))
    if random.random() < 0:
        scan_y = randomness(10, height - 10)
        scan_x = randomness(10, width - 10)
        intensity = random.uniform(0.5, 0.8)
        if random.random() > 0.5:
            mask_line[scan_x:scan_x + 6, :] = intensity
        else:
            mask_line[:, scan_y:scan_y + 6] = intensity
    mask_line = 1 - mask_line
    cell = cell * (mask_line) + (bg * (1 - (mask_line)))

    # TODO: Random line for invisibility & inside gradual transparancy using mask inner

    cell[cell < 0] = 0
    cell[cell > 255] = 255

    #########################################
    # Sometimes give background as cell     #
    #########################################
    if random.random() > 0.8:
        cell = bg
        label = np.zeros((width, height))

    #########################################
    # Generate segmentation labels          #
    #########################################
    if np.sum(label) != 0:
        label_gauss = generate_instancing_labels(label)
        label_gauss[label_gauss < 0] = 0
        label_gauss[label_gauss > 255] = 255
    else:
        label_gauss = label

    return cell, label, alt, label_gauss


def generate_sample_images(func):
    """
    Creates a 4x4 grid of a 16x16 grid for all outputs of the function provided.
    :param func: A cell creating function with cells, backgrounds, segmentation labels, and instancing labels as output.

    """
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
    grid_4 = torch.cat((torch.cat((torch.tensor(cell), torch.tensor(bg)), 1),
                        torch.cat((torch.tensor(label), torch.tensor(gauss)), 1)), 0)
    cv.imwrite("grid_4.png", grid_4.detach().numpy())


def generate_images(func, out_filename: str):
    """
    Creates the synthetic data dataset.

    :param func: A cell creating function with cells, backgrounds, segmentation labels, and instancing labels as output.
    :param out_filename: path and filename where data should be saved.
    """
    available_thread_count = cpu_count()
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

generate_sample_images(round_red_blood_cell)
# generate_sample_images(bullet_red_blood_cell)

# generate_images(round_red_blood_cell, prefix + "round_cells_new")
# generate_images(bullet_red_blood_cell, prefix + "bullet_cells_new")
