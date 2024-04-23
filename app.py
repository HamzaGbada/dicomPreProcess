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
import matplotlib.pyplot as plt
import uvicorn
from fastapi import UploadFile, File, FastAPI, HTTPException, Query

from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Methode, Data

app = FastAPI()
@app.post("/applyFedbs")
async def upload_file(
    file: UploadFile = File(...),
    method: Methode = Query(Methode.DOG, description="The method to be applied by default "),
    x: int = Query(..., description="X-coordinate of the center of the bounding box"),
    y: int = Query(..., description="Y-coordinate of the center of the bounding box")
):
    # Check if the uploaded file is a DICOM file
    if not file.filename.lower().endswith('.dcm'):
        raise HTTPException(status_code=400, detail="Only DICOM files are allowed")

    try:
        image = Data.load_dicom_image(file.file)
        array = image.array
        fedbs_array = Data.fedbs_main(Methode.DOG, array)
        array = PixelArrayOperation.getROI(array, x, y)
        plt.imshow(array)

        output = PixelArrayOperation.getROI(fedbs_array, x, y)
        plt.imshow(output)
        plt.show()

        return {"filename": file.filename, "content_type": "image/jpeg", "file": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


