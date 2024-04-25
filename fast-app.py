import uuid

import matplotlib.pyplot as plt
from fastapi import UploadFile, File, FastAPI, HTTPException, Query
from matplotlib import cm
from starlette.responses import FileResponse

from Mapper.mathOperation import PixelArrayOperation
from Service.Model import Data

app = FastAPI()


@app.post(
    "/applyFedbs",
    summary="Apply FEDBS Filter",
    description="Apply the Frequency Edge Detector Based Segmentation (FEDBS) filter to a DICOM image for micro-calcification detection. The FEDBS filter employs various methods such as Difference of Gaussian filter `dog`, Laplacian of Gaussian filter `log`, and Fourier filter `fft` for segmentation. For more information, see [Semanticscholar article](https://www.semanticscholar.org/paper/Mammograms-calcifications-segmentation-based-on-and/aa9eb94e808a7830635b940d6b566f1e2f965708).",
    responses={
        200: {
            "description": "Successfully processed image",
            "content": {"image/jpeg": {"example": "binary_data"}},
        },
        400: {
            "description": "Bad Request",
            "content": {"text/plain": {"example": "Only DICOM files are allowed"}},
        },
    },
)
async def applyFedbs(
    file: UploadFile = File(..., description="DICOM file to apply the FEDBS filter on"),
    method: str = Query(
        "dog",
        description="The method to apply: Difference of Gaussian filter `dog`, Laplacian of Gaussian filter `log`, or Fourier filter `fft`.",
    ),
    x: int = Query(
        ...,
        description="X-coordinate of the center of the bounding box for region of interest (ROI)",
    ),
    y: int = Query(
        ...,
        description="Y-coordinate of the center of the bounding box for region of interest (ROI)",
    ),
):
    """
    Apply the FEDBS filter to a DICOM image and return the processed image.

    Parameters:
    - **file**: DICOM file to apply the FEDBS filter on.
    - **method**: The method to apply by default is the Difference of Gaussian filter (dog),
                 supporting other methods like Laplacian of Gaussian filter (log) and Fourier filter (fft).
    - **x**: X-coordinate of the center of the bounding box for region of interest (ROI).
    - **y**: Y-coordinate of the center of the bounding box for region of interest (ROI).

    Returns:
    - **FileResponse**: Processed image in JPEG format.
    """

    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only DICOM files are allowed")

    image = Data.load_dicom_image(file.file)
    array = image.array

    fedbs_array = Data.fedbs_main(method, array)

    fedbs_roi_array = PixelArrayOperation.getROI(fedbs_array, x, y)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = f"./{filename}"
    plt.imsave(output_path, fedbs_roi_array, cmap=cm.gray)

    return FileResponse(output_path, media_type="image/jpeg", filename=filename)


@app.post(
    "/applyGrail",
    summary="Apply GRAIL Filter",
    description="Apply the GRAIL (Gabor-relying adjustment of image levels) filter to a DICOM image. The GRAIL filter is designed for automatic intensity windowing of mammographic images based on a perceptual metric. It converts a 12-bit image to an 8-bit image. For more information, see [AAPM article](https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.12144).",
    responses={
        200: {"description": "Successfully processed image", "content": {"image/jpeg"}},
        400: {
            "description": "Bad Request",
            "content": {"text/plain": {"example": "Only DICOM files are allowed"}},
        },
    },
)
async def applyGrail(
    file: UploadFile = File(..., description="DICOM file to apply the GRAIL filter on")
):
    """
    Apply the GRAIL filter to a DICOM image and return the processed image.

    Parameters:
    - **file**: DICOM file to apply the GRAIL filter on.

    Returns:
    - **FileResponse**: Processed image in JPEG format.
    """

    if not file.filename.lower().endswith(".dcm"):
        raise HTTPException(status_code=400, detail="Only DICOM files are allowed")

    image = Data.load_dicom_image(file.file)
    array = image.array

    output = Data.grail_main(array)

    filename = f"{uuid.uuid4()}.jpg"
    output_path = f"./{filename}"
    plt.imsave(output_path, output, cmap=cm.gray)

    return FileResponse(output_path, media_type="image/jpeg", filename=filename)
