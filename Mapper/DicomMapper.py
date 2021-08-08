#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import pydicom
import numpy as np
import pydicom._storage_sopclass_uids


class DicomMapper:
    def fromPixelToDicom(pixel_array, filename):
        """
        This method create a Dicom file from a numpy array
        INPUTS:
        pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
        filename: string name for the output file.
        """
        dRead = pydicom.dcmread(filename, force=True)
        uint16_image = np.uint16(pixel_array)
        dRead.PixelData = uint16_image.tobytes()
        dRead.save_as(filename)

        return

    def fromDicomToPixel(dicom_file):
        """
        This method extract pixel_array from Dicom file
        INPUTS:
        dicom_file: the Dicom file to be extracted
        :return:
        pixelData an uint16 numpy array
        """

        dRead = pydicom.dcmread(dicom_file, force=True)
        pixelData = dRead.pixel_array
        return pixelData