#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

from flask_restful import Resource
from flask import send_file


class DicomSend(Resource):
    def get(self):
        return send_file("temp.dcm")
