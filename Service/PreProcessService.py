import cv2
from Mapper.Normalization import Normalization
import logging as log
import numpy as np
from math import sqrt

class PreProcess:

    def OtsuThresholding(pixel_data, max):
        pixel_number = pixel_data.shape[0] * pixel_data.shape[1]
        mean_weight = 1.0 / pixel_number

        his, bins = np.histogram(pixel_data, np.arange(0, max+1))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(max)
        for t in bins[1:-1]:  # This goes from 1 to max-1 uint8 range (Pretty sure wont be those values)
            pc1 = np.sum(his[:t])
            pc2 = np.sum(his[t:])

            w1 = 1.0 / t
            w2 = 1.0 / (max - t)

            P1 = pc1 * mean_weight
            P2 = pc2 * mean_weight

            mu1 = pc1 / float(w1)
            mu2 = np.sum(intensity_arr[t:] * his[t:]) / float(w1)
            # print mub, muf
            sigma1 = np.sum(((intensity_arr[:t] * his[:t])-mu1) ** 2 ) / float(w1)
            sigma2 = np.sum(((intensity_arr[t:] * his[t:]) - mu2) ** 2) / float(w2)
            # value = P1 * P2 * (mu1 - mu2) ** 2
            value = P1*sigma1 + P2*sigma2
            value = sqrt(value)
            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = pixel_data.copy()
        # print('azertyuiop')
        # print(final_thresh)
        final_img[pixel_data > final_thresh] = max
        final_img[pixel_data < final_thresh] = 0
        return final_img

    def GammaBlur(self):
        """ Write your implementation here"""
        pass

    def ContrastChange(self):
        """ Write your implementation here"""
        pass