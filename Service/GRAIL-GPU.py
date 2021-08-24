from math import sin, cos, pi
from cmath import exp, sqrt
from operator import pow
import math
from numba import cuda
import numpy as np


class GaborGPU:
    @staticmethod
    @cuda.jit
    def gabor_blank_filter_gpu(kernel_size, scales, orientation, gabor_kernel):
        fmax = 0.25
        gamma = sqrt(2.0)
        eta = sqrt(2.0)
        x = cuda.threadIdx.x
        y = cuda.blockIdx.x
        i = cuda.blockIdx.y
        j = cuda.threadIdx.y
        b = pow(math.sqrt(2.0), i)
        fi = fmax / b
        alpha = fi / gamma
        beta = fi / eta
        theta = pi * (j / orientation)
        xprime = cos(theta) * ((x + 1) - ((kernel_size + 1) / 2)) + sin(theta) * ((y + 1) - ((kernel_size + 1) / 2))
        yprime = -sin(theta) * ((x + 1) - ((kernel_size + 1) / 2)) + cos(theta) * ((y + 1) - ((kernel_size + 1) / 2))
        gabor_kernel[i, j, x, y] = exp(-((pow(alpha, 2)) * (pow(xprime, 2)) + (pow(beta, 2)) * (pow(yprime, 2)))) * (
                fi ** 2 / (pi * gamma * eta)) * exp(fi * pi * xprime * 2j)

    @staticmethod
    @cuda.jit
    def convolve(result, mask, image):

        i, j = cuda.grid(2)
        image_rows, image_cols = image.shape

        delta_rows = mask.shape[0] // 2
        delta_cols = mask.shape[1] // 2

        s = 0 + 0j
        for k in range(mask.shape[0]):
            for l in range(mask.shape[1]):
                i_k = i - k + delta_rows
                j_l = j - l + delta_cols
                # (-4-) Check if (i_k, j_k) coordinates are inside the image:
                if (i_k >= 0) and (i_k < image_rows) and (j_l >= 0) and (j_l < image_cols):
                    s += mask[k, l] * image[i_k, j_l]
        result[i, j] = abs(s)

    @staticmethod
    @cuda.jit(device=True)
    def maximum_device(arr):
        max = arr[0, 0]
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] >= max:
                    max = arr[i, j]
        return max

    @staticmethod
    @cuda.jit(device=True)
    def minimum_device(x, limit):
        if (x > limit):
            return limit
        return x

    @staticmethod
    @cuda.jit
    def decomposition_normalization(feat_v):
        i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.x
        if i < 44 and j < 44:
            for k in range(18):
                feat_v[:, :, k][i, j] = feat_v[:, :, k][i, j] / GaborGPU.maximum_device(feat_v[:, :, k])
                feat_v[:, :, k][i, j] = GaborGPU.minimum_device(feat_v[:, :, k][i, j], 0.5) * 512

    @staticmethod
    @cuda.jit
    def feature_vect(gabor_result, d1, d2):
        gabor_result = gabor_result[::d1, ::d2]

    @staticmethod
    def gabor_decomposition_gpu(image):
        d1 = 1
        d2 = 1
        scales = 3
        orientation = 6
        # gabor kernel creation
        g_kernel = [[np.zeros((39, 39), np.complex128)] * 6 for i in range(3)]
        gabor_kernel = np.asarray(g_kernel)
        d_gabor_kernel = cuda.to_device(gabor_kernel)
        GaborGPU.gabor_blank_filter_gpu[(39, 3), (39, 6)](39, 3, 6, d_gabor_kernel)
        # applying the conv and setting feat_vect
        result = [[np.zeros(image.shape)] * 6 for i in range(3)]
        d_image = cuda.to_device(image)
        d_result = cuda.to_device(result)
        blockdim = (32, 32)
        griddim = (image.shape[0] // blockdim[0] + 1, image.shape[1] // blockdim[1] + 1)

        gabor_arr = [[np.zeros(image.shape)] * 6 for i in range(3)]
        d_gabor_arr = cuda.to_device(gabor_arr)

        for i in range(3):
            for j in range(6):
                GaborGPU.convolve[griddim, blockdim](d_result[i, j], d_gabor_kernel[i, j], d_image)
                GaborGPU.feature_vect[1, 1](d_result[i, j], d1, d2)
        d_result = d_result.reshape((image.shape[0] // d1, image.shape[1] // d2, scales * orientation), order='F')
        GaborGPU.decomposition_normalization[(2,), (32, 32)](d_result)
        k1 = d_result.copy_to_host()

        return k1

    def get_gabor(self):
        g_kernel = [[np.zeros((39, 39), np.complex128)] * 6 for i in range(3)]
        gabor_kernel = np.asarray(g_kernel)
        d_gabor_kernel = cuda.to_device(gabor_kernel)
        self.gabor_blank_filter_gpu[(39, 3), (39, 6)](39, 3, 6, d_gabor_kernel)
        kernel = d_gabor_kernel.copy_to_host()
        return kernel
