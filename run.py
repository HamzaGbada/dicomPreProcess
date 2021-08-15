from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data, Methode
import os
import logging

# Create and configure logger
logging.basicConfig(filename="grail.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":

    image = Data.load_dicom_image("20587054.dcm")
    array = image.array
    Data.plot_image(array)
    logger.debug("Pixel Data \n {}".format(array))
    logger.debug("Pixel Data shape \n {}".format(image.shape))
    image = PixelArrayOperation.getROI(array, 1250, 2000)
    # grail_array = Data.grail_main(array)
    fedbs_array = Data.fedbs_main(Methode.DOG, array)
    fedbs_image = PixelArrayOperation.getROI(fedbs_array, 1250, 2000)

    Data.plot_image(image)
    Data.plot_image(fedbs_image)