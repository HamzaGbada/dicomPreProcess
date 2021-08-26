#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np
from scipy import ndimage


class PixelArrayOperation:
    @staticmethod
    def morphoogy_closing(input):
        """
        Two-dimensional binary closing with the given structuring element.

        The *closing* of an input image by a structuring element is the
        *erosion* of the *dilation* of the image by the structuring element.
        For more information:
            https://en.wikipedia.org/wiki/Closing_%28morphology%29
            https://en.wikipedia.org/wiki/Mathematical_morphology

        Parameters:
            input : 2d ndarray
                Binary array_like to be closed. Non-zero (True) elements form
                the subset to be closed.

        Returns:
            binary_closing : ndarray of bools
                Closing of the input by the structuring element.

        Examples:
            >>> a = np.zeros((5,5), dtype=int)
            >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
            >>> a
            array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 0, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])
            >>> # Closing removes small holes
            >>> PixelArrayOperation.morphoogy_closing(a)
            array([[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])

            >>> a = np.zeros((7,7), dtype=int)
            >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
            >>> a
            array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
            >>> # In addition to removing holes, closing can also
            >>> # coarsen boundaries with fine hollows.
            >>> PixelArrayOperation.morphoogy_closing(a)
            array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]])
            """
        return ndimage.binary_closing(input, structure=np.ones((7, 7))).astype(np.int)

    @staticmethod
    def region_fill(input):
        """
        Fill the holes in binary images.
        For more:
            https://en.wikipedia.org/wiki/Mathematical_morphology
        Parameters:
            input : array_like
                2-D binary array with holes to be filled

        Returns:
            an ndarray
            Transformation of the initial image `input` where holes have been
            filled.

        Notes:
            The algorithm used in this function consists in invading the complementary
            of the shapes in `input` from the outer boundary of the image,
            using binary dilations. Holes are not connected to the boundary and are
            therefore not invaded. The result is the complementary subset of the
            invaded region.


        Examples:

        >>> a = np.zeros((5, 5), dtype=int)
        >>> a[1:4, 1:4] = 1
        >>> a[2,2] = 0
        >>> a
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
        >>> PixelArrayOperation.region_fill(a)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])

            """
        return ndimage.binary_fill_holes(input, structure=np.ones((7, 7))).astype(int)

    @staticmethod
    def fft(input):
        """
        This function calculates 2-dimensional discrete Fourier Transform using Fast Fourier Transform Algorithms (FFT)
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Fast_Fourier_transform

        Parameters:
            input: 2d ndarray to process.

        Returns:
            out: complex ndarray


        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (3,3))
            >>> fft = PixelArrayOperation.fft(a)
            >>> fft
            array([[19218.    +0.j        ,  1506. +1307.69835971j,
                     1506. -1307.69835971j],
                   [ 1455. +2527.06212824j,  2893.5 +995.06318895j,
                     1290.  +299.64478971j],
                   [ 1455. -2527.06212824j,  1290.  -299.64478971j,
                     2893.5 -995.06318895j]])

        """
        return np.fft.fft2(input)

    @staticmethod
    def inverse_fft(input):
        """
        This function calculates the inverse of 2-dimensional discrete Fourier Transform using Fast Fourier Transform Algorithms (FFT)
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Fast_Fourier_transform

        Parameters:
            input: 2d ndarray (it can be complex) to process.

        Returns:
            out: complex ndarray


        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (3,3))
            >>> ifft = PixelArrayOperation.inverse_fft(a)
            >>> ifft
            array([[19218.    +0.j        ,  1506. +1307.69835971j,
                     1506. -1307.69835971j],
                   [ 1455. +2527.06212824j,  2893.5 +995.06318895j,
                     1290.  +299.64478971j],
                   [ 1455. -2527.06212824j,  1290.  -299.64478971j,
                     2893.5 -995.06318895j]])

        """
        return np.fft.ifft2(input)

    @staticmethod
    def binarize(input, alpha):
        """
           This function binarize an image using local and global variance.
           For more infomation check this paper:
                https://www.researchgate.net/publication/306253912_Mammograms_calcifications_segmentation_based_on_band-pass_Fourier_filtering_and_adaptive_statistical_thresholding

           Parameters:
               input : a 2D ndarray.
               alpha : float
                    a scaling factor that relates the local and global variances.

           Returns:
               a 2D ndarray with the same size as the input containing 0 or 1 (a binary array)


           Examples:
                >>> a = np.random.randint(0, 5, (9,9))
                >>> a
                array([[3, 1, 0, 1, 1, 3, 4, 2, 2],
                       [0, 3, 3, 2, 3, 4, 0, 1, 1],
                       [0, 4, 4, 4, 3, 3, 1, 0, 3],
                       [4, 2, 3, 2, 2, 4, 2, 3, 4],
                       [2, 1, 3, 0, 0, 1, 4, 3, 1],
                       [2, 0, 0, 2, 0, 4, 0, 3, 1],
                       [4, 4, 4, 0, 4, 4, 1, 4, 2],
                       [2, 1, 3, 1, 2, 3, 1, 2, 0],
                       [4, 1, 3, 2, 3, 2, 3, 3, 0]])
                >>> PixelArrayOperation.binarize(a, 0.5)
                array([[1, 1, 0, 0, 1, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 1, 0, 0],
                       [1, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0]])

            """
        local_variance = PixelArrayOperation.getLocalVariance(input, 5)
        global_variance = PixelArrayOperation.variance(input)
        b = local_variance ** 2 < (alpha * global_variance ** 2)
        return np.where(b, 0, 1)

    @staticmethod
    def getROI(img, x, y, size=256):
        """
        This function return the region of interest in an image.
        """
        r = img[x - size:x + size + 1, y - size:y + size + 1]
        return r

    @staticmethod
    def variance(input):
        """
        Calculate the variance of the values of an 2-D image array

        Parameters:
            input: 2d ndarray to process.

        Returns:
            variance : float


        Examples:
            >>> a = np.array([[1, 2, 0, 0],
            ...               [5, 3, 0, 4],
            ...               [0, 0, 0, 7],
            ...               [9, 3, 0, 0]])
            >>> PixelArrayOperation.variance(a)
            7.609375
        """
        return ndimage.variance(input)

    @staticmethod
    def getLocalVariance(input, kernel):
        """
        Calculate the variance a specified sub-regions in image.

        Parameters:
            input: 2d ndarray to process.
            kernel: size of sub-region
        Returns:
            2D ndarray with the same size as the input contains the local variance of each region with size = kernel


        Examples:
            >>> a = np.random.randint(0, 5, (9,9))
            >>> PixelArrayOperation.getLocalVariance(a, 3)
            array([[0, 1, 1, 2, 1, 1, 1, 1, 2],
                   [0, 1, 2, 2, 1, 1, 1, 1, 1],
                   [0, 0, 1, 1, 2, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1, 1, 1, 1, 0],
                   [2, 1, 0, 1, 1, 2, 2, 2, 1],
                   [3, 2, 1, 0, 1, 2, 1, 1, 0],
                   [2, 2, 1, 0, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 1, 2, 1, 1],
                   [0, 0, 0, 0, 0, 1, 2, 2, 1]])

        """
        return ndimage.generic_filter(input, np.var, size=kernel)

    @staticmethod
    def butterworth_kernel(input, D_0=21, W=32, n=3):
        """
         Apply a Butterworth band-pass filter to enhance frequency features.
         This filter is defined in the Fourier domain.
         For more:
            https://en.wikipedia.org/wiki/Butterworth_filter

         Parameters:
             input: 2d ndarray to process.
             D_0: float
                  cutoff frequency
             W: float
                filter bandwidth
             n: int
                filter order
         Returns:
             band_pass: ndarray
                        The Butterworth-kernel.


         Examples:
             >>> a = np.random.randint(0, 5, (3,3))
             >>> PixelArrayOperation.butterworth_kernel(a)
             array([[0.78760795, 0.3821997 , 0.3821997 ],
                   [0.3821997 , 0.00479278, 0.00479278],
                   [0.3821997 , 0.00479278, 0.00479278]])

         """
        x = input.shape[0]
        y = input.shape[1]
        u, v = np.meshgrid(np.arange(x), np.arange(y))
        D = np.sqrt((u - x / 2) ** 2 + (v - y / 2) ** 2)
        band_pass = D ** 2 - D_0 ** 2
        cuttoff = 8 * W * D
        denom = 1.0 + (band_pass / cuttoff) ** (2 * n)
        band_pass = 1.0 / denom
        return band_pass.transpose()

    @staticmethod
    def make_step(delta, k_max):
        """
         This function compute the optimization grid.

         Parameters:
             delta: int
                    initial intensity span.
             k_max: int
                   number of iteration.
         Returns:
             steps: array with optimization steps.


         Examples:
             >>> PixelArrayOperation.make_step(300, 3)
             [300, 30, 3]

         """
        step_list = [delta]
        for s in range(k_max - 1):
            delta //= 10
            if delta % 1 != 0:
                break
            step_list.append(delta)
        return step_list

    @staticmethod
    def from12bitTo8bit(image, a, b):
        """
         This function resamples a 12 bit image to 8 bit outputting the resulting
         image as 12 bits. It applies lowest (a) and highest (b) intensity window.

         Parameters:
             delta: 2d ndarray to process.
             a: lowest intensity
             b: highest intensity

         Returns:
             a 12 bit 2d ndarray but resampled as if displayed in 8 bit


         Examples:
             >>> a = np.random.randint(0, 4095, (3,3))
             >>> a
             array([[2620, 2500,  881],
                   [1760, 2360,  634],
                   [1447, 2329,   93]])
             >>> PixelArrayOperation.from12bitTo8bit(a, 1997, 4080)
             array([[846.14509804, 707.40784314,  93.        ],
                   [ 93.        , 529.03137255,  93.        ],

         """
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
        """
        This function calculates Shannon Entropy of an image
        For more information about the Entropy this link:
        https://en.wikipedia.org/wiki/Entropy_(information_theory)

        Parameters:
            input: 2d ndarray to process.

        Returns:
            entropy: float rounded to 4 decimal places

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (512,512))
            >>> ent = InformationTheory.entropy(a)
            >>> ent
            11.9883

        """
        histogram, bin_edges = np.histogram(input, bins=int(input.max()) - int(input.min()) + 1,
                                            range=(int(input.min()), int(input.max()) + 1))
        probabilities = histogram / input.size
        probabilities = probabilities[probabilities != 0]

        return np.around(-np.sum(probabilities * np.log2(probabilities)), decimals=4)

    @staticmethod
    def joint_entropy(input, dest_data):
        """
        This function calculates joint entropy between two images.
        For more information about the joint entropy see this link:
        https://en.wikipedia.org/wiki/Joint_entropy

        Parameters:
            input: 2d ndarray.
            dest_data: 2d ndarray.

        Returns:
            joint entropy: float rounded to 4 decimal places

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (512,512))
            >>> b = np.random.randint(0, 4095, (512,512))
            >>> joint_ent = InformationTheory.joint_entropy(a,b)
            >>> joint_ent
            6.6435

        """
        joint_histogram, _, _ = np.histogram2d(input.flatten(), dest_data.flatten())
        joint_histogram_without_zero = joint_histogram[joint_histogram != 0]
        joint_prob = joint_histogram_without_zero / dest_data.size

        return np.around(-np.sum(joint_prob * np.log2(joint_prob)), decimals=4)

    @staticmethod
    def mutual_information(input, dest_data):
        """
        This function calculates mutual information between two images.
        For more information about the mutual information see this link:
        https://en.wikipedia.org/wiki/Mutual_information

        Parameters:
            input: 2d ndarray.
            dest_data: 2d ndarray.

        Returns:
            mi: float rounded to 4 decimal places.
                the mutual information between two images.

        Notes:
            The logarithm used is the bit logarithm (base-2).

        Examples:
            >>> import numpy as np
            >>> a = np.random.randint(0, 4095, (512,512))
            >>> b = np.random.randint(0, 4095, (512,512))
            >>> mi = InformationTheory.mutual_information(a,b)
            >>> mi
            17.3331
        """
        mi = InformationTheory.entropy(input) + InformationTheory.entropy(dest_data) - InformationTheory.joint_entropy(
            input, dest_data)

        return mi
