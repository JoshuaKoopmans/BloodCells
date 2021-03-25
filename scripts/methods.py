import imageio
from opensimplex import OpenSimplex
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2 as cv
import numpy as np
import random
import torch
import os
import scipy.stats as st

randomness = random.randint


def add_circle(image: np.ndarray, center: tuple, radius: int, color, thickness: int):
    return cv.circle(image, center=center, radius=radius, color=color, thickness=thickness)


def add_half_circle(image: np.ndarray, center: tuple, axes: tuple, angle: int, start_angle: int, end_angle: int,
                    color, thickness: int):
    return cv.ellipse(image, center=center, axes=axes, angle=angle, startAngle=start_angle, endAngle=end_angle,
                      color=color, thickness=thickness)


def elastic(image, alpha, sigma, seed=None):
    assert len(image.shape) == 2
    rs = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((rs.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


def gaussian_peak_from_label(label: np.ndarray):
    gauss2d = generate_instancing_labels(label)


def blur(image: np.ndarray, kernel=None):
    if kernel is None:
        kernel = (3, 3)
    return cv.GaussianBlur(image, kernel, 0)


def gaussian2d(kernel_lenght=21, sigma=8):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sigma, sigma, kernel_lenght + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    output = kern2d / kern2d.sum()
    return output / output.max()


def generate_instancing_labels(label: np.ndarray):
    height, width = label.shape[:2]
    gauss2d = gaussian2d(height)
    gauss2d = np.column_stack((gauss2d, np.zeros((gauss2d.shape[0], 25))))
    gauss2d = np.column_stack((np.zeros((gauss2d.shape[0], 25)), gauss2d))
    gauss2d = np.row_stack((gauss2d, np.zeros((25, gauss2d.shape[1]))))
    gauss2d = np.row_stack((np.zeros((25, gauss2d.shape[1])), gauss2d))
    x = torch.arange(0, 100).repeat(100, 1)
    y = x.T
    mat = torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), -1)
    white_is_better = mat[label == 255]
    x, y = torch.median(white_is_better, 0)[0]
    x_off = 50 - round(x.item())
    y_off = 50 - round(y.item())
    gauss_label = label.copy()
    gauss_label[label > 0] = (gauss2d[25 + y_off:125 + y_off, 25 + x_off:125 + x_off] * 255)[label > 0]
    # cv.imwrite("grass.png", gauss*255)
    return gauss_label


def add_background(image: np.ndarray):
    back_col = random.randint(20 - 10, 20 + 10)
    image[image < 5] = back_col
    return image


def warp_all_dim(image: np.ndarray, seed):
    dims = len(image.shape)
    if dims == 3:
        for d in range(dims):
            image[:, :, d] = elastic(image[:, :, d], 400, 15, seed=seed)
        return image
    else:
        return elastic(image, 400, 15, seed=seed)


def erode(image, kernel: np.ndarray, iterations=3):
    return cv.erode(src=image, kernel=kernel, iterations=iterations)


def dilate(image, kernel: tuple, iterations=3):
    unique = set(kernel)
    if len(unique) != 1:
        error_message = "Kernel input violation: left integer == right integer must be true!"
        raise Exception(error_message)
    elif unique.pop() % 2 == 0:
        error_message = "Kernel input violation: left integer = right integer % 2 != 0 must be true! " \
                        "Values must be uneven."
        raise Exception(error_message)
    kernel = np.ones((kernel[0], kernel[1]))
    return cv.dilate(src=image, kernel=kernel, iterations=iterations)


def create_bullet_cell(image, radius: int, x_offset: int, color, thickness: int, seed: int, memory=False):
    """

    :param image:
    :param radius:
    :param x_offset:
    :param color:
    :param thickness:
    :param seed:
    :param memory_seed:
    :param memory:
    :return:
    """

    # 180 at top, 360 at bottom, 090 at left, 270 right.

    # HALF CIRCLE PARAMETERS
    angle = 90
    start_angle = 180
    end_angle = 360

    random.seed(seed)

    # RANDOM VARIATIONS
    x_offset_variation = random.randint(x_offset - 5, x_offset + 5)
    first_circle_curve_variation = random.randint(-5, 3)
    second_circle_curve_variation = random.randint(-2, 1)
    connecting_line_variation_upper = random.randint(0, 5)
    connecting_line_variation_lower = random.randint(0, 5)
    radius_variation = random.randint(-4, 3)

    width, height = image.shape[:2]

    center_first = (int(width / 2), int(height / 2))
    center_second = (center_first[0] + x_offset_variation, center_first[1])
    axes = (int(radius), int(radius))
    axes_first = (axes[0], axes[1] - first_circle_curve_variation)
    axes_second = (axes[0] + radius_variation, axes[1] - second_circle_curve_variation)
    coordinates_first = ((center_first[0], center_first[1] - radius),
                         (center_second[0] - connecting_line_variation_upper,
                          center_second[1] - radius - radius_variation))
    coordinates_second = ((center_first[0], center_first[1] + radius),
                          (center_second[0] - connecting_line_variation_lower,
                           center_second[1] + radius + radius_variation))

    image = add_half_circle(image, center=center_first, axes=axes_first, angle=angle, start_angle=start_angle,
                            end_angle=end_angle, color=color, thickness=thickness)
    image = add_half_circle(image, center=center_second, axes=axes_second, angle=angle, start_angle=start_angle,
                            end_angle=end_angle, color=color, thickness=thickness)
    image = cv.line(image, coordinates_first[0], coordinates_first[1], thickness=thickness, color=color)
    image = cv.line(image, coordinates_second[0], coordinates_second[1], thickness=thickness, color=color)
    if memory:
        return image
    else:
        return image, seed


def genNoise(pixels, zoom=1, seed=-1):
    if seed == -1: seed = random.randint(0, 10e6)
    gen = OpenSimplex(seed=seed)
    pix = np.linspace(0, pixels * zoom, pixels)
    noi = np.array([[gen.noise2d(x, y) for x in pix] for y in pix])
    noi = noi.reshape((noi.shape[0], noi.shape[1], 1))
    return (noi - noi.min()) / (-noi.min() + noi.max())


def minmax(noise, min=0, max=1):
    return noise * (max - min) + min


def to_3d(image: np.ndarray):
    tmp = torch.zeros(image.shape[0], image.shape[1], 3)
    for i in range(3):
        tmp[:, :, i] = torch.from_numpy(image)
    return tmp.detach().numpy()


def interpolation(img1, img2, imap, mask=None):
    if type(mask) == type(None): mask = np.ones((img1.shape[0], img1.shape[1]))
    try:
        mask.shape[1]
    except:
        mask = minmax(mask.copy().astype(int), 0, 1)
    if len(mask.shape) == 2: mask = mask[:, :, np.newaxis]
    return (img1 * imap + img2 * (1 - imap)) * mask + img1 * (1 - mask)


def background_border(width: int, height: int, seed=1):
    border = np.zeros((width, height))
    mask = border.copy()

    y = randomness(10, 90)

    line_colors = (randomness(100 - 15, 110 + 30), randomness(0, 15), randomness(200, 255))
    mask_colors = (180, 255, 180)
    seed = int(random.random() * 10000)
    for i in range(len(line_colors)):
        line_thickness = randomness(1, 4)
        if line_colors[i] >= 180:
            line_thickness += randomness(0, 2)
        border[y:y + line_thickness, :] = line_colors[i]
        mask[y:y + line_thickness, :] = mask_colors[i]
        y += line_thickness

    ### ROTATE BORDERLINE ###
    if random.random() > 0.1:
        random_angle = randomness(0, 360)
        border = rotate_image(border, random_angle)
        mask = rotate_image(mask, random_angle)

    ### WARP BORDERLINE ###
    if random.random() > 0.4:
        if random.random() > 0.5:
            # BORDER WITH CURVE AT END#
            mask = elastic(mask, 700, 14, seed)
            border = elastic(border, 700, 14, seed)
        else:
            # NORMAL WARPING#
            mask = elastic(mask, 15, 3, seed)
            border = elastic(border, 15, 3, seed)
    border = blur(border, (5, 5))
    mask = blur(mask, (5, 5))

    mask = mask / 255.
    return border, mask


def rotate_image(image: np.ndarray, angle):
    # assert angle in range(0, 361, 1)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def background_floaters(width: int, height: int):
    center = (randomness(5, width - 5), randomness(5, height - 5))
    radius_outer = 2 + randomness(0, 3)
    radius_inner = radius_outer + randomness(-2, 0)
    if radius_inner == 0:
        radius_inner = 1
    if radius_inner >= radius_outer:
        radius_outer += 1
        radius_inner -= 1
    if random.random() < 0.5:
        color_outer = randomness(10, 30)
        color_inner = randomness(200, 255)
    else:
        color_outer = randomness(200, 255)
        color_inner = 33 + randomness(-5, 5)
    floater = np.zeros((width, height)) + 100
    floater = add_circle(image=floater, center=center,
                         radius=radius_outer, color=color_outer,
                         thickness=-1)
    floater = add_circle(image=floater, center=center,
                         radius=radius_inner, color=color_inner,
                         thickness=-1)
    mask = np.zeros((width, height))
    mask = add_circle(image=mask, center=center,
                      radius=radius_outer, color=100,
                      thickness=-1)
    mask = add_circle(image=mask, center=center,
                      radius=radius_inner, color=255,
                      thickness=-1)

    floater = blur(floater, (5, 5))
    mask = mask / 255

    return floater, mask


def base_background(width: int, height: int, seed: int, color_variation=(-35, 35), border_memory=None, mask_memory=None,
                    draw_border=True):
    background_color_variation = randomness(color_variation[0], color_variation[1])
    if border_memory is None or mask_memory is None:
        border, mask = background_border(width, height)
    else:
        border = border_memory
        mask = mask_memory
    ### BACKGROUND NOISE CLEAN ###
    background = minmax(genNoise(width, 1, seed), 100 + background_color_variation, 110 + background_color_variation)[:,
                 :, 0]
    background = blur(background, (3, 3))
    background += ((minmax(genNoise(width, 1, seed), -1, 1))[:, :, 0]) * 15
    background += ((minmax(genNoise(width, 0.01, seed), -1, 1))[:, :, 0]) * 10
    mask = mask * 255.
    mask[mask > 150] = 255
    mask[mask > 255] = 255

    if color_variation != (-35, 35):
        border[mask == 255] += background_color_variation

    border[border > 255] = 255
    border[border < 0] = 0
    mask = mask / 255.
    if not draw_border:
        mask = np.zeros((width, height))
    background = background * (1 - mask) + border * mask
    # color_indication = np.round(np.median(np.median(background, 0), 0))

    if (border_memory is not None) or (mask_memory is not None):
        return background
    else:
        return background, border, mask


def background(width: int, height: int, seed: int, alternative=False):
    if random.random() > 0.4:
        draw_borders = True
    else:
        draw_borders = False

    background, border, mask = base_background(100, 100, seed, draw_border=draw_borders)
    if random.random() > 0.5:
        alt_color_variation = (0, 50)
    else:
        alt_color_variation = (-50, 0)
    alt_background = base_background(100, 100, seed, alt_color_variation, border_memory=border, mask_memory=mask,
                                     draw_border=draw_borders)

    ## BLACK PIXEL GROUPS ON BACKGROUND ##
    for pixels in range(7):
        pixels = np.zeros((width, height))
        pixels = add_circle(pixels, (randomness(0, width), randomness(0, width)), randomness(1, 3), 255, -1)
        pixels = (255 - elastic(pixels, 300, 12)) / 255 + 0.1 + random.random() / 2
        pixels[pixels > 1] = 1
        background *= pixels
        alt_background *= pixels

    if random.random() > 0.1:
        # kleine zwarte en witte drolletjes
        if random.random() > 0.33:
            pixels, mask = background_floaters(width, height)
            background = background * (1 - mask) + pixels * mask


    if random.random() > 0.7:
        stripe = np.zeros((width, height))
        l = 25
        first = (randomness(0, 100 - l), randomness(0, 100 - l))
        sec = (first[0] + randomness(0, l), first[1] + randomness(0, l))
        stripe = cv.line(img=stripe, pt1=first, pt2=sec, color=255, thickness=randomness(1, 3))
        stripe = elastic(stripe, 500, 6)
        background *= (255 - stripe) / 255
        alt_background *= (255 - stripe) / 255
    if alternative:
        if random.random() > 0.1:
            # kleine zwarte en witte drolletjes
            pixels, mask = background_floaters(width, height)
            alt_background = alt_background * (1 - mask) + pixels * mask

    # background[background > 255] = 255
    # background[background < 0] = 0
    if alternative:
        return background, alt_background
    else:
        return background
