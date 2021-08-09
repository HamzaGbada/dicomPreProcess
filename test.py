import logging

from Mapper.mathOperation import PixelArrayOperation
from Service.GRAIL import Data
from PIL import Image, ImageOps
import numpy as np
# Create and configure logger

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


import matplotlib.pyplot as plt
import os

if os.path.exists("gabor_array.npz"):
    os.remove("gabor_array.npz")
g = Data("MR_small.dcm")
pixelData = g.pixel_data
logger.debug("Pixel Data \n {}".format(pixelData))
logger.debug("Pixel Data shape \n {}".format(pixelData.shape))
image2 = np.asarray(ImageOps.grayscale(Image.open("samir.jpg")))
# image = PixelArrayOperation.getROI(pixelData, 1250, 2000)
# x1 = g.grail_main()
#
x = g.fedbs_main(image2,0)

plt.imshow(image2,cmap=plt.cm.gray)
plt.show()
# plt.imshow(x1,cmap=plt.cm.gray)
# plt.show()
plt.imshow(x,cmap=plt.cm.gray)
plt.show()
