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

    image = Data.load_dicom_image("MR_small.dcm")
    array = image.array
    Data.plot_image(array)
    logger.debug("Pixel Data \n {}".format(array))
    logger.debug("Pixel Data shape \n {}".format(image.shape))

    grail_array = Data.grail_main(array)
    fedbs_array = Data.fedbs_main(Methode.DOG, array)

    Data.plot_image(fedbs_array)