#  بكي صاحبي لما رأى الدرب دونه *** و أيقن أنا لاحقان بقيصرا
#  فقلت له لا تبكي عينك إنما *** نحاول ملكا أو نموت فنعذرا
# إمرؤ القيس

import pydicom
import enum
import matplotlib.pyplot as plt
from scipy import ndimage
import logging
from Service.GRAIL import Gabor_information
from Mapper.mathOperation import PixelArrayOperation
from Service.PreProcessService import PreProcess
import numpy as np

logging.basicConfig(
    filename="grail.log", format="%(asctime)s %(message)s", filemode="w"
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Bits(enum.Enum):
    BIN = 2
    OCTAT = 8
    SYLLABLE = 12
    HEXA = 16


class Format(enum.Enum):
    GRAY = 1
    RGB = 3



class Image:
    def __init__(self, array, array_bits):
        self._array = array
        if len(array.shape) < 3 or array.shape[2] == 1:
            self._array_format = Format.GRAY
        elif len(array.shape) == 3:
            self._array_format = Format.RGB

        self._shape = array.shape
        self._data_type = array.dtype
        self._array_bits = array_bits

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, array):
        self._array = array

    @property
    def array_format(self):
        return self._array_format

    @array_format.setter
    def array_format(self, array_format):
        self._array_format = array_format

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @property
    def array_bits(self):
        return self._array_bits

    @array_bits.setter
    def array_bits(self, array_bits):
        self._array_bits = array_bits


class Data:
    @staticmethod
    def load_dicom_image(path):
        dicom_reader = pydicom.dcmread(path, force=True)
        pixel_data = dicom_reader.pixel_array
        return Image(pixel_data, Bits.SYLLABLE)

    @staticmethod
    def plot_image(image):
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()

    @staticmethod
    def grail_main(array):
        """
        This function is the main Grail algorithm. It finds the min and max intensity value
        of a mammogram based on a perceptual metric which combines mutual information
        and Gabor filtering.
        For more about GRAIL algorithm check this:
            https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12144
        Parameters:
            array: 2d ndarray to process.

        Returns:
           pixel_data: the new array after quality adjustment.
        """
        logger.debug("Pixel Data In Main \n {}".format(array))
        a, b = Gabor_information.get_best_a_b(array)
        logger.debug("Best a and b \n {}  \n {}".format(a, b))
        WL = 0.5 * (b - a)
        WW = b - a
        L = 0.5 * (WL - WW)
        H = 0.5 * (WL + WW)
        pixel_data = PixelArrayOperation.from12bitTo8bit(array, L, H)
        return pixel_data

    @staticmethod
    def fedbs_main(method_type, array):
        """
        This function is the main FEDBS algorithm. It binarize mammogram to make
        micro-calcification appear.
        For more about FEDBS algorithm check this:
            https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12144
        Parameters:
            method_type: Enum
            array: 2d ndarray to process.

        Returns:
           b_f: a binary array.
        """
        if method_type == "dog":
            sigma1 = 1.7
            sigma2 = 2
            fi = PreProcess.DoG(array, sigma1, sigma2)
        elif method_type == "log":
            sigma = 2
            fi = PreProcess.LoG(array, sigma)
        elif method_type == "fft":
            froi = PixelArrayOperation.fft(array)
            H = PixelArrayOperation.butterworth_kernel(froi)
            fi = PixelArrayOperation.inverse_fft(np.multiply(froi, H))
        fi = ndimage.median_filter(abs(fi), size=(5, 5))
        fi = PreProcess.GammaCorrection(fi, 1.25)
        if method_type == "fft":
            B = PixelArrayOperation.binarize(fi, 1)
        else:
            B = PreProcess.OtsuThresholding(fi)
        I = PixelArrayOperation.morphoogy_closing(B)
        b_f = PixelArrayOperation.region_fill(I)
        return b_f
