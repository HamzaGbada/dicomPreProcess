import logging

from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data
from args import argument

# Logger Setup
logging.basicConfig(
    filename="grail.log", format="%(asctime)s %(message)s", filemode="w"
)

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    # x,y are the coordinates of the central point in ROI
    x = 1250
    y = 2000
    image = Data.load_dicom_image("data/" + argument.file)
    array = image.array
    logger.debug("Pixel Data \n {}".format(array))
    logger.debug("Pixel Data shape \n {}".format(image.shape))
    if argument.algorithm == "grail":
        output = Data.grail_main(array)
    if argument.algorithm == "fedbs":
        if argument.method == "dog":
            fedbs_array = Data.fedbs_main("dog", array)
            array = PixelArrayOperation.getROI(array, x, y)
            output = PixelArrayOperation.getROI(fedbs_array, x, y)
        if argument.method == "log":
            fedbs_array = Data.fedbs_main("log", array)
            array = PixelArrayOperation.getROI(array, x, y)
            output = PixelArrayOperation.getROI(fedbs_array, x, y)
        if argument.method == "fft":
            array = PixelArrayOperation.getROI(array, x, y)
            output = Data.fedbs_main("fft", array)

    Data.plot_image(array)
    Data.plot_image(output)
