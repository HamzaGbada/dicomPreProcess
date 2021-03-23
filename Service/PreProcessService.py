import numpy as np
from PIL import Image,ImageEnhance


class PreProcess:

    def OtsuThresholding(self, max):
        """This function calculates the Otsu thresholding of the image in the Dicom file.
            For more information about the Otsu method see this link:
            https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            self: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the thresholded image.

        """

        # Number of pixels in an image
        pixel_number = self.shape[0] * self.shape[1]

        mean_weight = 1.0 / pixel_number

        # Histogram Calculation
        his, bins = np.histogram(self, np.arange(0, max + 1))

        # Initialization
        final_thresh = -1
        final_value = -1

        for t in bins[1:-1]:

            pc1 = np.sum(his[:t])
            pc2 = np.sum(his[t:])

            w1 = 1.0 / t
            w2 = 1.0 / (max - t)

            P1 = pc1 * mean_weight
            P2 = pc2 * mean_weight

            # Mean class calculation
            mu1 = np.sum(his[:t]) / float(w1)
            mu2 = np.sum(his[t:]) / float(w2)

            # Interclass variance
            value = P1 * P2 * (mu1 - mu2) ** 2

            # Determination of the threshold
            if value > final_value:
                final_thresh = t
                final_value = value

        final_img = self.copy()

        # Binarization
        final_img[self > final_thresh] = max
        final_img[self < final_thresh] = 0

        return final_img

    def GammaCorrection(self, gamma):
        """This function calculates the Gamma Correction of the image in the Dicom file.
            For more information about the Gamma Correction see this link:
            https://en.wikipedia.org/wiki/Gamma_correction

        Args:
            self: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the thresholded image.

        """
        image = np.power(self/float(np.max(self)), gamma)

        return image

    def ContrastAdjust(self, contrast, brightness):
        """ Write your implementation here"""
        # final_img = self.copy()
        # final_img[final_img < final_img.max()-brightness ] += brightness
        # return final_img
        img = self.copy()
        quotient = self.max()//2
        img = img * (contrast / quotient + 1) - contrast
        img = np.clip(img, 0, self.max())
        return img

