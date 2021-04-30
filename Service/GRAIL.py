#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري
import numpy as np
from Mapper.mathOperation import PixelArrayOperation


class GRAIL:

    @classmethod
    def gabor_kernel(self, kernel_size, sigma, gamma, lamda, psi, angle):
        gabor_kernel = np.zeros((kernel_size, kernel_size), np.float32)
        m = kernel_size // 2
        n = kernel_size // 2
        # degree -> radian
        theta = angle / 180. * np.pi
        for i in range(-m, m + 1):
            for j in range(-n, n + 1):
                # get kernel x
                x = np.cos(theta) * i + np.sin(theta) * j
                # get kernel y
                y = -np.sin(theta) * i + np.cos(theta) * j

                gabor_kernel[i + m, j + n] = np.exp(-(x ** 2 + gamma ** 2 * y ** 2) / (2 * sigma ** 2)) * np.cos(
                    2 * np.pi * x / lamda + psi)

        return gabor_kernel

    def gabor_response(pixel_data, kernel_size, sigma, gamma, lamda, psi, theta):

        kernel = GRAIL.gabor_kernel(kernel_size, sigma, gamma, lamda, psi, theta)
        im_filtered = np.zeros(pixel_data.shape, dtype=np.float32)
        im_filtered[:, :] = PixelArrayOperation.convolution(pixel_data[:, :], kernel)

        return im_filtered

    def quality_measurement(self):
        pass

    def get_best_A_B(self):
        pass
