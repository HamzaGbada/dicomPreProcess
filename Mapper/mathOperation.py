#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np


class PixelArrayOperation:

    def make_step(delta, k_max):
        step_list = []
        step_list.append(delta)
        for s in range(k_max - 1):
            delta //= 10
            if delta % 1 != 0:
                break
            step_list.append(delta)
        return step_list
    
    def from12bitTo8bit(pixel_data, a, b):
        maxvalue = 255
        maximum = int(pixel_data.max())
        minimum = int(pixel_data.min())
        if a == b:
            normed = (pixel_data - a) / a * maxvalue
        else:
            normed = (pixel_data - a) / (b - a) * maxvalue
        round_array = np.rint(normed)
        m = np.clip(round_array, 0, maxvalue)
        image8 = m / maxvalue * (maximum - minimum) + minimum
        return image8

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

class InformationTheory:

    @classmethod
    def entropy(self, pixel_data):
        maximum = int(pixel_data.max())
        minimum = int(pixel_data.min())
        size = pixel_data.size
        histogram, bin_edges = np.histogram(pixel_data, bins=maximum - minimum + 1, range=(minimum, maximum + 1))
        probabilities = histogram / size
        probabilities = probabilities[probabilities != 0]
        log2 = np.log2(probabilities)

        return -np.sum(probabilities * log2)

    @classmethod
    def joint_entropy(self, pixel_data, dest_data):

        pixel_data = pixel_data.flatten()
        dest_data = dest_data.flatten()
        joint_histogram, _, _ = np.histogram2d(pixel_data, dest_data)
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / dest_data.size

        return -np.sum(joint_prob * np.log2(joint_prob))

    def mutual_information(pixel_data, dest_data):

        mi = InformationTheory.entropy(pixel_data) + InformationTheory.entropy(dest_data) - InformationTheory.joint_entropy(pixel_data, dest_data)

        return mi

    def mutual_information_12_and_8bit(pixel_data, a, b):
        octat_array = PixelArrayOperation.from12bitTo8bit(pixel_data, a, b)
        octat_mi = InformationTheory.mutual_information(pixel_data, octat_array)

        return octat_mi


