import uuid

import matplotlib.pyplot as plt
from fastapi import UploadFile, File, FastAPI, HTTPException, Query
from matplotlib import cm
from starlette.responses import FileResponse

from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data

app = FastAPI()

@app.post("/applyFedbs", responses={
    200: {
        "content": {"image/jpeg": {}}
    }
})
async def applyFedbs(
        file: UploadFile = File(...),
        method: str = Query("dog", description="The method to be applied by default"),
        x: int = Query(..., description="X-coordinate of the center of the bounding box"),
        y: int = Query(..., description="Y-coordinate of the center of the bounding box")
):
    if not file.filename.lower().endswith('.dcm'):
        raise HTTPException(status_code=400, detail="Only DICOM files are allowed")

    image = Data.load_dicom_image(file.file)
    array = image.array
    fedbs_array = Data.fedbs_main(method, array)

    roi_array = PixelArrayOperation.getROI(array, x, y)

    fedbs_roi_array = PixelArrayOperation.getROI(fedbs_array, x, y)



    filename = f"{uuid.uuid4()}.jpg"
    output_path = f"./{filename}"
    plt.imsave(output_path, fedbs_roi_array, cmap=cm.gray)



    return FileResponse(output_path, media_type='image/jpeg', filename=filename)