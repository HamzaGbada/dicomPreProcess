#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري
import logging
import numpy as np
from math import sqrt, pi
import pydicom
from scipy import ndimage
from skimage.filters import gabor

from Mapper.mathOperation import PixelArrayOperation
from Mapper.mathOperation import InformationTheory

# Create and configure logger
from Service.PreProcessService import PreProcess

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


class Gabor:
    def __init__(self):
        self._pixel_array_operation = PixelArrayOperation()

    def gabor_blank_filter(self, image, scales, orientation):
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
                filt_real, filt_imag = gabor(image, fi, theta, sigma_x=alpha, sigma_y=beta)
                gabor_out = np.absolute(filt_real + 1j * filt_imag)
                gabor_array[i][j] = gabor_out
        return np.array(gabor_array)

    def gabor_feature(self, input, scales, orientations, d1, d2):
        input = input.astype(float)
        feature_vector = np.empty(0)
        gabor_abs = self.gabor_blank_filter(input, 3, 6)
        for i in range(scales):
            for j in range(orientations):
                feature_vector = np.append(feature_vector, gabor_abs[i,j,::d1, ::d2].reshape(-1))
        return np.reshape(feature_vector,(input.shape[0] // d1, input.shape[1] // d2, scales*orientations), order='F')

    def gabor_decomposition(self, input, scales, orientations, kernel_size=39, d1=1, d2=1):
        feature_size = scales * orientations
        feat_v = self.gabor_feature(input, scales, orientations, d1, d2)
        for i in range(feature_size):
            max_feat = np.max(feat_v[:, :, i])
            if max_feat != 0.0:
                feat_v[:, :, i] = feat_v[:, :, i] / np.max(feat_v[:, :, i])
        feat_v = np.minimum(feat_v, 0.5) * 512

        return feat_v

    def gabor_8bit_respresentation(self, input, a, b, scales, orientations):
        octat_array = self._pixel_array_operation.from12bitTo8bit(input, a, b)
        octat_gabor = self.gabor_decomposition(octat_array, scales, orientations)

        return octat_gabor


class Gabor_information(Gabor, InformationTheory):

    def gabor_mutual_information(self, input, gabor_pixel_data, a, b, scales, orientations):
        octat_gabor = self.gabor_8bit_respresentation(input, a, b, scales, orientations)
        gabor_mi = self.mutual_information(gabor_pixel_data, octat_gabor)

        return gabor_mi

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
        pixel_data_gabor = self.gabor_decomposition(input, scales, orientations)

        mutual_info_array = np.empty(0)

        for b_k in b_step:
            logger.debug("b_k in highest intensity \n {}".format(b_k))
            gabor_mi = self.gabor_mutual_information(input, pixel_data_gabor, a_k, b_k, scales, orientations)
            logger.debug("gabor Mutual Information in highest intensity \n {}".format(gabor_mi))
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
        pixel_data_gabor = self.gabor_decomposition(input, scales, orientations)

        mutual_info_array = np.empty(0)

        for a_k in a_step:
            logger.debug("a_k in lowest intensity \n {}".format(a_k))
            gabor_mi = self.gabor_mutual_information(input, pixel_data_gabor, b_k, a_k, scales, orientations)
            logger.debug("gabor Mutual Information in lowest intensity \n {}".format(gabor_mi))
            mutual_info_array = np.append(mutual_info_array, gabor_mi)

        return mutual_info_array, a_step

    def get_best_a_b(self, input, scales=3, orientations=6, delta=300, k_max=3):
        step_list = self._pixel_array_operation.make_step(delta, k_max)
        logger.debug("Step List in get_best_a_b \n {} ".format(step_list))
        b_0 = input.max()  # bmax
        b_mean = round(np.mean(input))  # bmin
        a_0 = input.min()  # amin
        a_mean = round(np.mean(input))  # amax

        logger.debug("bmax before update \n {}".format(b_0))
        logger.debug("bmin before update \n {}".format(b_mean))
        logger.debug("amin before update \n {}".format(a_0))
        logger.debug("amax before update \n {}".format(a_mean))
        a_init = a_0
        b_init = b_0
        for step in step_list:
            logger.debug("step during update  \n {}".format(step))
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
            if (a_init == best_a and b_init == best_b):
                break
            a_init = best_a
            b_init = best_b
            a_0 = max(best_a - step, 0)
            a_mean = best_a + step
            b_mean = best_b - step
            b_0 = max(best_b + step, input.max())
            logger.debug("bmax during update \n {}".format(b_0))
            logger.debug("bmin during update \n {}".format(b_mean))
            logger.debug("amin during update \n {}".format(a_0))
            logger.debug("amax during update \n {}".format(a_mean))

        return best_a, best_b


class Data:
    def __init__(self, path):
        self._path = path
        dicom_reader = pydicom.dcmread(path, force=True)
        self._pixel_data = dicom_reader.pixel_array
        self._gabor_information = Gabor_information()
        self._pixel_array_operation = PixelArrayOperation()
        self._preprocess = PreProcess()

    @property
    def pixel_data(self):
        return self._pixel_data

    @pixel_data.setter
    def pixel_data(self, pixel_data):
        self._pixel_data = pixel_data

    def grail_main(self):
        logger.debug("Pixel Data In Main \n {}".format(self._pixel_data))
        a, b = self._gabor_information.get_best_a_b(self._pixel_data)
        logger.debug("Best a and b \n {}  \n {}".format(a, b))
        WL = 0.5 * (b - a)
        WW = b - a
        L = 0.5 * (WL - WW)
        H = 0.5 * (WL + WW)
        pixel_data = self._pixel_array_operation.from12bitTo8bit(self._pixel_data, L, H)
        self.pixel_data = pixel_data
        return pixel_data

    def fedbs_main(self, method_type):
        if method_type == 0:
            sigma1 = 2
            sigma2 = 1.7
            dog = self._preprocess.DoG(sigma1, sigma2)
            fi = ndimage.correlate(self._pixel_data, dog, mode='constant')
        elif method_type == 1:
            sigma = 2
            log = self._preprocess.LoG(sigma)
            fi = ndimage.correlate(self._pixel_data, log, mode='constant')
        else:
            froi = self._pixel_array_operation.fft(self._pixel_data)
            H = self._pixel_array_operation.butterworth_kernel(froi)
            fi = self._pixel_array_operation.inverse_fft(froi * H)
        fi = ndimage.median_filter(abs(fi), size=(5, 5))
        fi = self._preprocess.GammaCorrection(fi, 1.25)
        B = self._pixel_array_operation.binarize(fi, 1)
        I = self._pixel_array_operation.morphoogy_closing(B)
        b_f = self._pixel_array_operation.region_fill(I)
        return b_f
