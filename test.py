import logging
from Service.GRAIL import Data

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

x1 = g.grail_main()

x = g.fedbs_main(0)

plt.imshow(pixelData,cmap=plt.cm.gray)
plt.show()
plt.imshow(x1,cmap=plt.cm.gray)
plt.show()
plt.imshow(x,cmap=plt.cm.gray)
plt.show()
