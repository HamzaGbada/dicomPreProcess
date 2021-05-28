#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np
import scipy


class PixelArrayOperation:

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

    def convolution(oldimage, kernel):
        return scipy.ndimage.correlate(oldimage, kernel, mode='constant')


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
        mi = self.entropy(input) + self.entropy(dest_data) - self.joint_entropy(input, dest_data)

        return mi

    def mutual_information_12_and_8bit(self, input, a, b):
        octat_array = PixelArrayOperation.from12bitTo8bit(input, a, b)
        octat_mi = self.mutual_information(input, octat_array)

        return octat_mi
