from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data, Methode
import logging

logging.basicConfig(filename="grail.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    var = input("give algorithm name:\n")
    image = Data.load_dicom_image("data/20587054.dcm")
    array = image.array
    logger.debug("Pixel Data \n {}".format(array))
    logger.debug("Pixel Data shape \n {}".format(image.shape))
    if var == 'grail':
        output = Data.grail_main(array)
    if var == 'fedbs' :
        method = input('give methode name:\n')
        if method == 'dog':
            fedbs_array = Data.fedbs_main(Methode.DOG, array)
            array = PixelArrayOperation.getROI(array, 1250, 2000)
            output = PixelArrayOperation.getROI(fedbs_array, 1250, 2000)
        if method == 'log':
            fedbs_array = Data.fedbs_main(Methode.LOG, array)
            array = PixelArrayOperation.getROI(array, 1250, 2000)
            output = PixelArrayOperation.getROI(fedbs_array, 1250, 2000)
        if method == 'bbp':
            array = PixelArrayOperation.getROI(array, 1250, 2000)
            output = Data.fedbs_main(Methode.FFT, array)

    Data.plot_image(array)
    Data.plot_image(output)
