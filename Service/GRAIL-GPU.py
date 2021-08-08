from math import sin, cos, pi
from cmath import exp, sqrt
from operator import pow
from numba import cuda
import numpy as np
import math


class GaborGPU:
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

    def get_gabor(self):
        g_kernel = [[np.zeros((39, 39), np.complex128)] * 6 for i in range(3)]
        gabor_kernel = np.asarray(g_kernel)
        d_gabor_kernel = cuda.to_device(gabor_kernel)
        self.gabor_blank_filter_gpu[(39, 3), (39, 6)](39, 3, 6, d_gabor_kernel)
        kernel = d_gabor_kernel.copy_to_host()
        return kernel
