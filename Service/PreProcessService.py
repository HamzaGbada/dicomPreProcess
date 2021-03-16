from abc import ABC, abstractmethod

class Preprocess(ABC):

    @abstractmethod
    def OtsuThresholding(self):
        pass

    @abstractmethod
    def GammaBlur(self):
        pass

    @abstractmethod
    def ContrastChange(self):
        pass
