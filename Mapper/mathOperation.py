#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np
from scipy import ndimage
from skimage.filters import butterworth

class PixelArrayOperation:
    @staticmethod
    def morphoogy_closing(b):
        return ndimage.binary_closing(b, structure=np.ones((7, 7))).astype(np.int)

    @staticmethod
    def region_fill(b):
        return ndimage.binary_fill_holes(b, structure=np.ones((7, 7))).astype(int)

    @staticmethod
    def fft(img):
        return np.fft.fft2(img)

    @staticmethod
    def inverse_fft(img):
        return np.fft.ifft2(img)

    @staticmethod
    def binarize(img, alpha):
        local_variance = PixelArrayOperation.getLocalVariance(img, 5)
        global_variance = PixelArrayOperation.variance(img)
        b = local_variance ** 2 < (alpha * global_variance ** 2)
        return np.where(b, 0, 1)

    @staticmethod
    def getROI(img, x, y, size=256):
        r = img[x - size:x + size + 1, y - size:y + size + 1]
        return r

    @staticmethod
    def variance(img):
        return ndimage.variance(img)

    @staticmethod
    def getLocalVariance(oldimage, kernel):
        return ndimage.generic_filter(oldimage, np.var, size=kernel)

    # @staticmethod
    # def butterworth_kernel(img, D_0=21, W=32, n=3):
    #     x = img.shape[0]
    #     y = img.shape[1]
    #     u, v = np.meshgrid(np.arange(x), np.arange(y))
    #     D = np.sqrt((u - x / 2) ** 2 + (v - y / 2) ** 2)
    #     band_pass = D ** 2 - D_0 ** 2
    #     cuttoff = W * D
    #     denom = 1.0 + (band_pass / cuttoff) ** (2 * n)
    #     band_pass = 1.0 / denom
    #     return band_pass.transpose()
    @staticmethod
    def butterworth_kernel(img, D_0=21, W=32, n=3):
        high = butterworth(img, D_0, True, order=n)
        low = butterworth(high, W, False, order=n)
        return low

    @staticmethod
    def make_step(delta, k_max):
        step_list = [delta]
        for s in range(k_max - 1):
            delta //= 10
            if delta % 1 != 0:
                break
            step_list.append(delta)
        return step_list

    @staticmethod
    def from12bitTo8bit(image, a, b):
        if a == b:
            if a == 0:
                normed = image * 255
            else:
                normed = (image - a) / a * 255
        else:
            normed = (image - a) / (b - a) * 255

        return (np.clip(np.rint(normed), 0, 255)) / 255 * (int(image.max()) - int(image.min())) + int(image.min())


class InformationTheory:
    @staticmethod
    def entropy(input):
        histogram, bin_edges = np.histogram(input, bins=int(input.max()) - int(input.min()) + 1,
                                            range=(int(input.min()), int(input.max()) + 1))
        probabilities = histogram / input.size
        probabilities = probabilities[probabilities != 0]

        return np.around(-np.sum(probabilities * np.log2(probabilities)), decimals=4)

    @staticmethod
    def joint_entropy(input, dest_data):
        joint_histogram, _, _ = np.histogram2d(input.flatten(), dest_data.flatten())
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / dest_data.size

        return np.around(-np.sum(joint_prob * np.log2(joint_prob)), decimals=4)

    @staticmethod
    def mutual_information(input, dest_data):
        mi = InformationTheory.entropy(input) + InformationTheory.entropy(dest_data) - InformationTheory.joint_entropy(input, dest_data)

        return mi
