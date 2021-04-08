#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري


class ImageModel(object):

    def __init__(self, width, height, pixel_data):
        self.__width = width
        self.__height = height
        self.__pixel_data = pixel_data

    @property
    def height(self):
        return self.__height
    @property
    def width(self):
        return self.__width
    @property
    def pixel_data(self):
        return self.__pixel_data

    @width.setter
    def width(self, width):
        self.__width = width

    @height.setter
    def height(self, height):
        self.__height = height

    @pixel_data.setter
    def pixel_data(self, pixel_data):
        self.__pixel_data = pixel_data