#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np
from skimage.filters import gaussian, difference_of_gaussians, threshold_otsu, threshold_sauvola


class PreProcess:
    @staticmethod
    def OtsuThresholding(pixel_array, output=None):
        """This function calculates the Otsu thresholding of the image in the Dicom file.
            For more information about the Otsu method see this link:
            https://en.wikipedia.org/wiki/Otsu%27s_method

        Args:
            pixel_array: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the threshold image.

        """
        if output == None:
            output = np.empty_like(pixel_array)
        x = threshold_otsu(pixel_array)
        output[pixel_array < x] = 0
        output[pixel_array > x] = 1

        return output

    @staticmethod
    def GammaCorrection(pixel_array, gamma):
        """This function calculates the Gamma Correction of the image in the Dicom file.
            For more information about the Gamma Correction see this link:
            https://en.wikipedia.org/wiki/Gamma_correction

        Args:
            pixel_array: pixel_array in the Dicom file.
            max: Maximum value that can be assigned to a pixel.

        Returns:
            The return value is the corrected image.

        """
        image = np.power(pixel_array / float(4095), gamma)
        image = image * 4095

        return image

    @staticmethod
    def ContrastAdjust(pixel_array, contrast, brightness, output=None):
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
            pixel_array: pixel_array in the Dicom file.
            contrast: the contrast value to be applied.
            brightness: the brightness value to be applied.

        Returns:
            The return value is an image with different contrast and brightness.

        """
        if output == None:
            output = pixel_array.copy()
        quotient = 4095 // 2
        alpha = contrast / quotient + 1
        beta = brightness - contrast
        output = output * alpha + beta

        output = np.clip(output, 0, 4095)
        return output

    @staticmethod
    def DoG(img, sigma1, sigma2):
        return difference_of_gaussians(img, sigma1, sigma2)

    @staticmethod
    def LoG(img, sigma):
        return difference_of_gaussians(img, sigma)

    @staticmethod
    def sauvola(image, gamma):
        return threshold_sauvola(image, window_size=15, k=gamma)
