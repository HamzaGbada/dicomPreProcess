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
g = Data("test.png")
pixelData = g.get_pixel_data()
logger.debug("Pixel Data \n {}".format(pixelData))
logger.debug("Pixel Data shape \n {}".format(pixelData.shape))

x = g.main()
plt.imshow(pixelData,cmap=plt.cm.gray)
plt.show()
plt.imshow(x,cmap=plt.cm.gray)
plt.show()