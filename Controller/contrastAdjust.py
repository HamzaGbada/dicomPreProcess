from flask_restful import Resource
from Controller.Model import ImageModel
from Service.PreProcessService import PreProcess
from flask import request
import numpy as np
import json
from Mapper.jsonMapper import NumpyArrayEncoder

class ContrastAdjust(Resource):

    def post(self, contrast, brightness):
        imageModel = ImageModel(None,None,None);
        data = request.get_json(force=True)
        imageList = data["pixel_data"]
        image = np.array(imageList)
        output = PreProcess.ContrastAdjust(image, contrast, brightness)
        imageModel.pixel_data = output
        imageModel.height = output.shape[0]
        imageModel.width = output.shape[1]

        # Serialization
        numpyData = {"array": imageModel.pixel_data}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)

        return encodedNumpyData