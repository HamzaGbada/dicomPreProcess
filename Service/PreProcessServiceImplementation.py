from Service.PreProcessService import Preprocess
import cv2
import logging as log

class PreProcessImpl(Preprocess):

    def OtsuThresholding(self, pixel_data):
        """ Write your implementation here"""
        if len(pixel_data.shape) != 1 :
            log.error("the image should be in grayscale")
            return 1
        ret, th = cv2.threshold(pixel_data, 0, 65535, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pass

    def GammaBlur(self):
        """ Write your implementation here"""
        pass

    def ContrastChange(self):
        """ Write your implementation here"""
        pass