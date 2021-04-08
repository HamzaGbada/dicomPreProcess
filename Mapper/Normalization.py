#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import numpy as np


class Normalization:
    def normalize(pixel_data):
        x = 255/4096
        matrix = x*pixel_data
        return matrix.astype(np.uint8)

    def denormailze(pixel_data):
        x = 4096 / 255
        matrix = x * pixel_data
        return matrix.astype(np.uint16)