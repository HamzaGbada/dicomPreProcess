# from flask import Flask, render_template
# from flask_restful import Api
# from Controller.fileSend import DicomSend
# from Controller.otsuthreshold import OtsuThreshold
# from Controller.gammaCorrection import GammaCorrection
# from Controller.contrastAdjust import ContrastAdjust
# from flask_cors import CORS
#
# app = Flask(__name__)
# api = Api(app)
# CORS(app)
#
#
# @app.route("/")
# def index():
#     return render_template("page_1/index.html")
#
#
# api.add_resource(DicomSend, "/dicomFile/metadata")
# api.add_resource(GammaCorrection, "/gammaCorrection/<float:gamma>")
# api.add_resource(OtsuThreshold, "/otsuThreshold/<int:max>")
# api.add_resource(ContrastAdjust, "/contrastAdjust/<int:contrast>/<int:brightness>")
#
# if __name__ == "__main__":
#     app.run(port=5000, debug=True)
import os
import uuid

import fastapi
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
from fastapi import UploadFile, File, FastAPI, HTTPException, Query
from fastapi.openapi.models import Response
from starlette.responses import FileResponse

from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data

app = FastAPI()

@app.post("/applyFedbs", responses={
    200: {
        "content": {"image/jpeg": {}}
    }
})
async def upload_file(
        file: UploadFile = File(...),
        method: str = Query("dog", description="The method to be applied by default"),
        x: int = Query(..., description="X-coordinate of the center of the bounding box"),
        y: int = Query(..., description="Y-coordinate of the center of the bounding box")
):
    # Check if the uploaded file is a DICOM file
    if not file.filename.lower().endswith('.dcm'):
        raise HTTPException(status_code=400, detail="Only DICOM files are allowed")

    try:
        image = Data.load_dicom_image(file.file)
        array = image.array
        fedbs_array = Data.fedbs_main(method, array)

        # Get the region of interest from the original image array
        roi_array = PixelArrayOperation.getROI(array, x, y)

        # Get the region of interest from the processed image array
        fedbs_roi_array = PixelArrayOperation.getROI(fedbs_array, x, y)

        # Convert the processed image array to an image
        fedbs_image = Image.fromarray(fedbs_roi_array)

        # Save the image as a JPEG file
        filename = f"{uuid.uuid4()}.jpg"
        output_path = f"./{filename}"
        print(f"fedbs images type {type(fedbs_image)}")
        print(f"fedbs images  : \n{fedbs_image}")
        fedbs_image.save(output_path, format="JPEG")

        # Read the saved JPEG file
        with open(output_path, "rb") as file:
            file_data = file.read()

        # Remove the saved JPEG file
        # os.remove(output_path)
        if not output_path.is_file():
            return {"error": "Image not found on the server"}

        return FileResponse("001.dcm", media_type='image/jpeg', filename="001.dcm")


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

