#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np


class PixelArrayOperation:

    def normalize(self):
        x = 255 / 4096
        matrix = x * self
        return matrix.astype(np.uint8)

    def denormalize(self):
        x = 4096 / 255
        matrix = x * self
        return matrix.astype(np.uint16)

    def convolution(self):
        pass


class InformationTheory:

    @classmethod
    def entropy(self, pixel_data):
        maximum = pixel_data.max()
        minimum = pixel_data.min()
        size = pixel_data.size
        histogram, bin_edges = np.histogram(pixel_data, bins=maximum - minimum + 1, range=(minimum, maximum + 1))
        probabilities = histogram / size
        probabilities = probabilities[probabilities != 0]
        log2 = np.log2(probabilities)

        return -np.sum(probabilities * log2)

    @classmethod
    def joint_entropy(self, pixel_data, dest_data):
        joint_histogram, _, _ = np.histogram2d(pixel_data, dest_data)
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / dest_data.size

        return -np.sum(joint_prob * np.log2(joint_prob))

    def mutual_information(pixel_data, dest_data):

        mi = InformationTheory.entropy(pixel_data) + InformationTheory.entropy(dest_data) - InformationTheory.joint_entropy(pixel_data, dest_data)

        return mi
