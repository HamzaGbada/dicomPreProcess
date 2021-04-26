#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np


class PreProcess:

    def OtsuThresholding(self, max):
        """This function calculates the Otsu thresholding of the image in the Dicom file.
            For more information about the Otsu method see this link:
            https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            self: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the threshold image.

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
            The return value is the corrected image.

        """
        image = np.power(self/float(4095), gamma)
        image = image*4095

        return image

    def ContrastAdjust(self, contrast, brightness):
        """This function used to adjust the contrast and the brightness
        the image in the Dicom file.
            This approach is based this equation that can be used to apply
        both contrast and brightness at the same time:
            ****
            new_img = alpha*old_img + beta
            ****
            Where alpha and beta are contrast and brightness coefficient respectively:

            alpha 1  beta 0      --> no change
            0 < alpha < 1        --> lower contrast
            alpha > 1            --> higher contrast
            -2047 < beta < +2047   --> good range for brightness values

            In my case:
                alpha = contrast / 2047 + 1
                beta = brightness - contrast

        Args:
            self: pixel_array in the Dicom file.
            contrast: the contrast value to be applied.
            brightness: the brightness value to be applied.

        Returns:
            The return value is an image with different contrast and brightness.

        """
        img = self.copy()
        quotient = 4095//2
        alpha = contrast / quotient + 1
        beta = brightness - contrast
        img = img * alpha + beta

        img = np.clip(img, 0, 4095)
        return img

    def contrast_approch(self):
        pass