#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np
from scipy import ndimage



class PixelArrayOperation:

    def morphoogy_closing(self, b):
        return ndimage.binary_closing(b, structure=np.ones((7, 7))).astype(np.int)

    def region_fill(self, b):
        return ndimage.binary_fill_holes(b, structure=np.ones((7, 7))).astype(int)

    def fft(self, img):
        return np.fft.fft2(img)

    def inverse_fft(self, img):
        return np.fft.ifft2(img)

    def binarize(self, img, alpha):
        local_variance = self.getLocalVariance(img, 5)
        global_variance = self.variance(img)
        b = local_variance ** 2 < (alpha * global_variance ** 2)
        return np.where(b, 0, 1)

    def getROI(self, img, x, y, size=256):
        r = img[x - size:x + size + 1, y - size:y + size + 1]
        return r

    def variance(self, img):
        return ndimage.variance(img)

    def getLocalVariance(self, oldimage, kernel):
        return ndimage.generic_filter(oldimage, np.var, size=kernel)

    def butterworth_kernel(self, img, D_0=21, W=32, n=3):
        m = img.shape[0]
        n = img.shape[1]
        u = np.arange(m)
        v = np.arange(n)
        u, v = np.meshgrid(u, v)
        D = np.sqrt((u - m / 2) ** 2 + (v - n / 2) ** 2)
        band_pass = D ** 2 - D_0 ** 2
        cuttoff = W * D
        denom = 1.0 + (band_pass / cuttoff) ** (2 * n)
        band_pass = 1.0 / denom
        return band_pass.transpose()

    def make_step(self, delta, k_max):
        step_list = []
        step_list.append(delta)
        for s in range(k_max - 1):
            delta //= 10
            if delta % 1 != 0:
                break
            step_list.append(delta)
        return step_list

    def from12bitTo8bit(self, image, a, b):
        if a == b:
            if a == 0:
                normed = image * 255
            else:
                normed = (image - a) / a * 255
        else:
            normed = (image - a) / (b - a) * 255

        return (np.clip(np.rint(normed), 0, 255)) / 255 * (int(image.max()) - int(image.min())) + int(image.min())


class InformationTheory:
    def entropy(self, input):
        histogram, bin_edges = np.histogram(input, bins=int(input.max()) - int(input.min()) + 1,
                                            range=(int(input.min()), int(input.max()) + 1))
        probabilities = histogram / input.size
        probabilities = probabilities[probabilities != 0]

        return np.around(-np.sum(probabilities * np.log2(probabilities)), decimals=4)

    def joint_entropy(self, input, dest_data):
        joint_histogram, _, _ = np.histogram2d(input.flatten(), dest_data.flatten())
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / dest_data.size

        return np.around(-np.sum(joint_prob * np.log2(joint_prob)), decimals=4)

    def mutual_information(self, input, dest_data):
        input_f = input.flatten()
        dest_f = input.flatten()
        metrics.mutual_info_score(input_f, dest_f)
        mi = self.entropy(input) + self.entropy(dest_data) - self.joint_entropy(input, dest_data)

        return mi
