#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري
import numpy as np
from Mapper.mathOperation import PixelArrayOperation
from math import sqrt, pi


class GRAIL:

    @classmethod
    def gabor_kernel(self, kernel_size, f, theta):

        gabor_kernel = np.zeros((kernel_size, kernel_size), dtype='complex_')
        m = kernel_size // 2
        n = kernel_size // 2
        for k in range(-m, m + 1):
            for l in range(-n, n + 1):

                x = np.cos(theta) * k + np.sin(theta) * l
                y = -np.sin(theta) * k + np.cos(theta) * l

                gabor_kernel[k + m - 1, l + n - 1] = np.exp(-(x ** 2 + y ** 2) * (1 / 2 * f ** 2)) * np.exp(f * np.pi * x * 2j)

        return gabor_kernel

    @classmethod
    def gabor_blank_filter(self, kernel_size, scales, orientation):
        fmax = 0.25
        gabor_list = []
        for i in range(scales):
            fi = fmax / (sqrt(2) ** i)
            for j in range(orientation):
                theta = pi * (j / orientation)
                gabor_list.append(GRAIL.gabor_kernel(kernel_size, fi, theta))
        return gabor_list

    @classmethod
    def gabor_feature(self, pixel_data, gabor_list, d1, d2, scale, orientation):
        gabor_result = []
        for kernel in gabor_list:
            im_filtered = np.zeros(pixel_data.shape, dtype='complex_')
            im_filtered[:, :] = PixelArrayOperation.convolution(pixel_data[:, :], kernel)
            gabor_result.append(im_filtered)
        feature_vector = np.empty(0)
        for res in gabor_result:
            gabor_abs = abs(res)
            feat_col = gabor_abs[::d1, ::d2].reshape(-1)
            feature_vector = np.append(feature_vector, feat_col)
        return feature_vector

    def gabor_response(pixel_data, kernel_size, f, theta):

        kernel = GRAIL.gabor_kernel(kernel_size, f, theta)
        im_filtered = np.zeros(pixel_data.shape, dtype=np.float32)
        im_filtered[:, :] = PixelArrayOperation.convolution(pixel_data[:, :], kernel)

        return im_filtered

    def quality_measurement(self):
        pass

    def get_best_A_B(self):
        pass
