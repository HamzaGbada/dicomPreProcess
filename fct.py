#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري
import logging
import scipy.ndimage
import numpy as np
from math import sqrt, pi

import pydicom

from Mapper.mathOperation import PixelArrayOperation
from Mapper.mathOperation import InformationTheory

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


def gabor_kernel(kernel_size, f, theta):
    gabor_kernel = np.zeros((kernel_size, kernel_size), dtype='complex_')
    m = kernel_size // 2
    n = kernel_size // 2
    for k in range(-m, m + 1):
        for l in range(-n, n + 1):
            x = np.cos(theta) * k + np.sin(theta) * l
            y = -np.sin(theta) * k + np.cos(theta) * l

            gabor_kernel[k + m - 1, l + n - 1] = np.exp(-(x ** 2 + y ** 2) * (1 / 2 * f ** 2)) * np.exp(
                f * np.pi * x * 2j)

    return gabor_kernel


def gabor_blank_filter(kernel_size, scales, orientation):
    fmax = 0.25
    gamma = sqrt(2)
    eta = sqrt(2)
    gabor_array = [[np.empty(0)] * orientation for i in range(scales)]
    for i in range(scales):
        fi = fmax / (sqrt(2) ** i)
        alpha = fi / gamma
        beta = fi / eta
        for j in range(orientation):
            theta = pi * (j / orientation)
            gabor_kernel = np.zeros((kernel_size, kernel_size), dtype='complex_')
            for x in range(kernel_size):
                for y in range(kernel_size):
                    xprime = np.cos(theta) * ((x + 1) - ((kernel_size + 1) / 2)) + np.sin(theta) * (
                                (y + 1) - ((kernel_size + 1) / 2))
                    yprime = -np.sin(theta) * ((x + 1) - ((kernel_size + 1) / 2)) + np.cos(theta) * (
                                (y + 1) - ((kernel_size + 1) / 2))
                    gabor_kernel[x, y] = np.around(np.exp(-((alpha ** 2) * (xprime ** 2) + (beta ** 2) * (yprime ** 2))) * (
                                fi ** 2 / (pi * gamma * eta)) * np.exp(fi * np.pi * xprime * 2j), decimals = 15)
            gabor_array[i][j] = gabor_kernel
    return gabor_array


def gabor_feature(input, gabor_list, d1, d2):
    # THIS METHOD IS TRUE
    input = input.astype(float)
    u = len(gabor_list)
    v = len(gabor_list[0])
    gabor_result = [[np.empty(0)] * v for i in range(u)]
    for i in range(u):
        for j in range(v):
            gabor_result[i][j] = scipy.ndimage.correlate(input, gabor_list[i][j], mode='constant')
    feature_vector = np.empty(0)
    for i in range(u):
        for j in range(v):
            gabor_abs = abs(gabor_result[i][j])
            feature_vector = np.append(feature_vector, gabor_abs[::d1, ::d2].reshape(-1))
    return feature_vector


def gabor_decomposition(input, scales, orientations, kernel_size=39, d1=1, d2=1):
    feature_size = scales * orientations
    gabor_list = gabor_blank_filter(kernel_size, scales, orientations)
    feature_vector = gabor_feature(input, gabor_list, d1, d2)
    feat_v = np.reshape(feature_vector,
                        (input.shape[0]//d1, input.shape[1]//d2, feature_size), order='F')
    for i in range(feature_size):
        max_feat = np.max(feat_v[:, :, i])
        if max_feat != 0.0:
            feat_v[:, :, i] = feat_v[:, :, i] / np.max(feat_v[:, :, i])
    feat_v = np.minimum(feat_v, 0.5) * 512

    return feat_v


def gabor_8bit_respresentation(input, a, b, scales, orientations):
    octat_array = PixelArrayOperation.from12bitTo8bit(input, a, b)
    octat_gabor = gabor_decomposition(octat_array, scales, orientations)

    return octat_gabor


def gabor_response(kernel_size, f, theta):
    im_filtered = np.zeros(input.shape, dtype='complex_')
    im_filtered[:, :] = PixelArrayOperation.convolution(input[:, :], gabor_kernel(kernel_size, f, theta))

    return im_filtered


def gabor_mutual_information(input, gabor_pixel_data, a, b, scales, orientations):
    octat_gabor = gabor_8bit_respresentation(input, a, b, scales, orientations)
    gabor_mi = mutual_information(gabor_pixel_data, octat_gabor)

    return gabor_mi


def gabor_entropy(input, gabor_pixel_data, a, b, scales, orientations):
    return joint_entropy(gabor_pixel_data, gabor_8bit_respresentation(input, a, b, scales, orientations))


def mutual_information_gabor_highest_intensity(input, step, scales, orientations, b_0=None, b_mean=None,
                                               a_0=None):
    if b_0 is None:
        b_0 = input.max()
    if a_0 is None:
        a_0 = input.min()
    if b_mean is None:
        b_mean = round(np.mean(input))

    a_k = a_0
    b_step = np.arange(b_0, b_mean - 1, -step)
    pixel_data_gabor = gabor_decomposition(input, scales, orientations, 2, 1, 1)

    mutual_info_array = np.empty(0)

    for b_k in b_step:
        logger.debug("b_k in highest intensity \n {}".format(b_k))
        gabor_mi = gabor_mutual_information(input, pixel_data_gabor, a_k, b_k, scales, orientations)
        logger.debug("gabor Mutual Information in highest intensity \n {}".format(gabor_mi))
        mutual_info_array = np.append(mutual_info_array, gabor_mi)

    return mutual_info_array, b_step


def mutual_information_gabor_lowest_intensity(input, step, scales, orientations, a_mean=None, a_0=None,
                                              b_0=None):
    if b_0 is None:
        b_0 = input.max()
    if a_0 is None:
        a_0 = input.min()
    if a_mean is None:
        a_mean = round(np.mean(input))

    b_k = b_0
    a_step = np.arange(a_0, a_mean + 1, step)
    pixel_data_gabor = gabor_decomposition(input, scales, orientations, 2, 1, 1)

    mutual_info_array = np.empty(0)

    for a_k in a_step:
        logger.debug("a_k in lowest intensity \n {}".format(a_k))
        gabor_mi = gabor_mutual_information(input, pixel_data_gabor, b_k, a_k, scales, orientations)
        logger.debug("gabor Mutual Information in lowest intensity \n {}".format(gabor_mi))
        mutual_info_array = np.append(mutual_info_array, gabor_mi)

    return mutual_info_array, a_step


def get_best_a_b(input, scales=3, orientations=6, delta=300, k_max=3):
    step_list = PixelArrayOperation.make_step(delta, k_max)
    logger.debug("Step List in get_best_a_b \n {} ".format(step_list))
    b_0 = input.max()  # bmax
    b_mean = round(np.mean(input))  # bmin
    a_0 = input.min()  # amin
    a_mean = round(np.mean(input))  # amax

    logger.debug("bmax before update \n {}".format(b_0))
    logger.debug("bmin before update \n {}".format(b_mean))
    logger.debug("amin before update \n {}".format(a_0))
    logger.debug("amax before update \n {}".format(a_mean))
    for step in step_list:
        logger.debug("step during update  \n {}".format(step))
        mutual_info_right_array, b_step = mutual_information_gabor_highest_intensity(input, step, scales,
                                                                                     orientations, b_0, b_mean,
                                                                                     a_0)
        max_ind = np.argmax(mutual_info_right_array)
        best_b = b_step[max_ind]

        mutual_info_left_array, a_step = mutual_information_gabor_lowest_intensity(input, step, scales,
                                                                                   orientations, a_mean, a_0,
                                                                                   best_b)
        max_ind = np.argmax(mutual_info_left_array)
        best_a = a_step[max_ind]

        a_0 = max(best_a - step, 0)
        a_mean = best_a + step
        b_mean = best_b - step
        b_0 = max(best_b + step, input.max())
        logger.debug("bmax during update \n {}".format(b_0))
        logger.debug("bmin during update \n {}".format(b_mean))
        logger.debug("amin during update \n {}".format(a_0))
        logger.debug("amax during update \n {}".format(a_mean))

    return best_a, best_b


def make_step(delta, k_max):
    step_list = []
    step_list.append(delta)
    for s in range(k_max - 1):
        delta //= 10
        if delta % 1 != 0:
            break
        step_list.append(delta)
    return step_list


def from12bitTo8bit(image, a, b):
    if a == b:
        if a == 0:
            normed = image * 255
        else:
            normed = (image - a) / a * 255
    else:
        normed = (image - a) / (b - a) * 255

    return (np.clip(np.rint(normed), 0, 255)) / 255 * (int(image.max()) - int(image.min())) + int(image.min())


def convolution(oldimage, kernel):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if (len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage,
                           pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2), (0, 0)),
                           mode='constant', constant_values=0).astype(complex)
    elif (len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=(
            (kernel_h // 2, kernel_h // 2), (kernel_w // 2, kernel_w // 2)), mode='constant',
                           constant_values=0).astype(complex)
    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape, dtype='complex_')

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]


def entropy(input):
    histogram, bin_edges = np.histogram(input, bins=int(input.max()) - int(input.min()) + 1,
                                        range=(int(input.min()), int(input.max()) + 1))
    probabilities = histogram / input.size
    probabilities = probabilities[probabilities != 0]

    return -np.sum(probabilities * np.log2(probabilities))


def joint_entropy(input, dest_data):
    joint_histogram, _, _ = np.histogram2d(input.flatten(), dest_data.flatten())
    joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
    joint_prob = joint_histogram_without_zero / dest_data.size

    return -np.sum(joint_prob * np.log2(joint_prob))


def mutual_information(input, dest_data):
    mi = entropy(input) + entropy(dest_data) - joint_entropy(input, dest_data)

    return mi


def mutual_information_12_and_8bit(input, a, b):
    octat_array = PixelArrayOperation.from12bitTo8bit(input, a, b)
    octat_mi = mutual_information(input, octat_array)

    return octat_mi
