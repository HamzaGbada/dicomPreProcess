#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري
import numpy as np
from math import sqrt, pi

import pydicom

from Mapper.mathOperation import PixelArrayOperation
from Mapper.mathOperation import InformationTheory
from PIL import Image


class Gabor:
    def gabor_kernel(self, kernel_size, f, theta):

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

    def gabor_blank_filter(self, kernel_size, scales, orientation):

        fmax = 0.25
        gabor_list = []
        for i in range(scales):
            fi = fmax / (sqrt(2) ** i)
            for j in range(orientation):
                theta = pi * (j / orientation)
                gabor_list.append(self.gabor_kernel(kernel_size, fi, theta))
        return gabor_list

    def gabor_feature(self, input, gabor_list, d1, d2):
        gabor_result = []
        for kernel in gabor_list:
            im_filtered = np.zeros(input.shape, dtype='complex_')
            im_filtered[:, :] = PixelArrayOperation.convolution(input[:, :], kernel)
            gabor_result.append(im_filtered)
        feature_vector = np.empty(0)
        for res in gabor_result:
            gabor_abs = abs(res)
            feature_vector = np.append(feature_vector, gabor_abs[::d1, ::d2].reshape(-1))
        return feature_vector

    def gabor_decomposition(self, input, scales, orientations, kernel_size=39, d1=1, d2=1):
        feature_size = scales * orientations
        gabor_list = self.gabor_blank_filter(kernel_size, scales, orientations)
        feat_v = np.reshape(self.gabor_feature(input, gabor_list, d1, d2),
                            (input.shape[0], input.shape[1], feature_size), order='F')
        for i in range(feature_size):
            max_feat = np.max(feat_v[:, :, i])
            if max_feat != 0.0:
                feat_v[:, :, i] = feat_v[:, :, i] / np.max(feat_v[:, :, i])
        feat_v = np.minimum(feat_v, 0.5) * 512

        return feat_v

    def gabor_8bit_respresentation(self, input, a, b, scales, orientations):
        octat_array = PixelArrayOperation.from12bitTo8bit(input, a, b)
        octat_gabor = self.gabor_decomposition(octat_array, scales, orientations)

        return octat_gabor

    def gabor_response(self, kernel_size, f, theta):

        im_filtered = np.zeros(input.shape, dtype='complex_')
        im_filtered[:, :] = PixelArrayOperation.convolution(input[:, :], self.gabor_kernel(kernel_size, f, theta))

        return im_filtered


class Gabor_information(Gabor, InformationTheory):

    def gabor_mutual_information(self, input, gabor_pixel_data, a, b, scales, orientations):
        octat_gabor = self.gabor_8bit_respresentation(input, a, b, scales, orientations)
        gabor_mi = self.mutual_information(gabor_pixel_data, octat_gabor)

        return gabor_mi

    def gabor_entropy(self, input, gabor_pixel_data, a, b, scales, orientations):

        return self.joint_entropy(gabor_pixel_data, self.gabor_8bit_respresentation(input, a, b, scales, orientations))

    def mutual_information_gabor_highest_intensity(self, input, step, scales, orientations, b_0=None, b_mean=None,
                                                   a_0=None):

        if b_0 is None:
            b_0 = input.max()
        if a_0 is None:
            a_0 = input.min()
        if b_mean is None:
            b_mean = round(np.mean(input))

        a_k = a_0
        b_step = np.arange(b_0, b_mean - 1, -step)
        pixel_data_gabor = self.gabor_decomposition(input, scales, orientations, 2, 1, 1)

        mutual_info_array = np.empty(0)

        for b_k in b_step:
            gabor_mi = self.gabor_mutual_information(input, pixel_data_gabor, a_k, b_k, scales, orientations)
            mutual_info_array = np.append(mutual_info_array, gabor_mi)

        return mutual_info_array, b_step

    def mutual_information_gabor_lowest_intensity(self, input, step, scales, orientations, a_mean=None, a_0=None,
                                                  b_0=None):

        if b_0 is None:
            b_0 = input.max()
        if a_0 is None:
            a_0 = input.min()
        if a_mean is None:
            a_mean = round(np.mean(input))

        b_k = b_0
        a_step = np.arange(a_0, a_mean + 1, step)
        pixel_data_gabor = self.gabor_decomposition(input, scales, orientations, 2, 1, 1)

        mutual_info_array = np.empty(0)

        for a_k in a_step:
            gabor_mi = self.gabor_mutual_information(input, pixel_data_gabor, b_k, a_k, scales, orientations)
            mutual_info_array = np.append(mutual_info_array, gabor_mi)

        return mutual_info_array, a_step

    def get_best_a_b(self, input, scales=3, orientations=6, delta=300, k_max=3):

        step_list = PixelArrayOperation.make_step(delta, k_max)

        b_0 = input.max()  # bmax
        b_mean = round(np.mean(input))  # bmin
        a_0 = input.min()  # amin
        a_mean = round(np.mean(input))  # amax
        for step in step_list:
            mutual_info_right_array, b_step = self.mutual_information_gabor_highest_intensity(input, step, scales,
                                                                                              orientations, b_0, b_mean,
                                                                                              a_0)
            max_ind = np.argmax(mutual_info_right_array)
            best_b = b_step[max_ind]

            mutual_info_left_array, a_step = self.mutual_information_gabor_lowest_intensity(input, step, scales,
                                                                                            orientations, a_mean, a_0,
                                                                                            best_b)
            max_ind = np.argmax(mutual_info_left_array)
            best_a = a_step[max_ind]

            a_0 = max(best_a - step, 0)
            a_mean = best_a + step
            b_mean = best_b - step
            b_0 = max(best_b + step, input.max())

        return best_a, best_b


class Data(Gabor_information):
    def __init__(self, path):
        self._path = path
        dicom_reader = pydicom.dcmread(path,force=True)
        self._pixel_data = dicom_reader.pixel_array

    def get_pixel_data(self):
        return self._pixel_data

    def main(self):
        a, b = self.get_best_a_b(self._pixel_data)
        pixel_data = np.where(self._pixel_data > b, 255, self._pixel_data)
        pixel_data = np.where(pixel_data < a, 0, pixel_data)
        pixel_data = np.where(np.logical_and(pixel_data <= b, pixel_data >= a), (pixel_data - a) / (b - a) * 255,
                              pixel_data)
        return pixel_data
