#  أليلتنا بذي حسم انيري *** إذا انت انقضيت فلا تحوري
#  فإن يكن بالذنائب طال ليلي *** فقد ابكي من الليل القصيري

import werkzeug
from flask_restful import Resource, reqparse
from Service.PreProcessService import PreProcess
import numpy as np
from flask import render_template, make_response
from Mapper.DicomMapper import DicomMapper


class ContrastAdjust(Resource):
    def __init__(self):
        self._preprocess = PreProcess()

    def post(self, contrast, brightness):
        # get the dicom file from post request
        parse = reqparse.RequestParser()
        parse.add_argument(
            "file", type=werkzeug.datastructures.FileStorage, location="files"
        )
        args = parse.parse_args()
        uploaded_file = args["file"]
        uploaded_file.save("temp.dcm")

        # get pixel_array from the dicom file
        imageList = DicomMapper.fromDicomToPixel("temp.dcm")

        # process the pixel_data
        image = np.array(imageList)
        output = self._preprocess.ContrastAdjust(image, contrast, brightness)

        # save the processed image to the dicom file
        DicomMapper.fromPixelToDicom(output, "temp.dcm")

        # render the html template
        headers = {"Content-Type": "text/html"}
        return make_response(render_template("page_2/index.html"), 200, headers)
