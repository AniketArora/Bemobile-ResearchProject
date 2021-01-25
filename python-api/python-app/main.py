import numpy as np
import json
from typing import List
from io import BytesIO

import time
import uvicorn
import orjson
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, validator

from api_functions import ApiFunctions
from helpers import check_shape_or_raise

# initialise needed data
class NumpyBody(BaseModel):
    Array: np.ndarray

    @validator('Array', pre=True)
    def parse_values(cls, v):
        try:
            v = json.loads(v)
            return np.array(v, dtype=float)
        except:
            raise ValueError('Unable to parse input. Are you sure its a serialized numpy array?')

    class Config:
        arbitrary_types_allowed = True

# This fixes https://github.com/tiangolo/fastapi/issues/459
class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


print("Started Api server")
start = time.time()
app = FastAPI(default_response_class=ORJSONResponse)
end = time.time()
print(f"Api server started, time elapsed: {end - start}")

print("Started loading model")
start = time.time()
functions = ApiFunctions()
end = time.time()
print(f"Model Loaded, time elapsed: {end - start}")

@app.get("/")
async def main():
    content = """
<body>
<form action="/predict/file" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)


@app.post("/predict/file")
async def predict_from_file(files: List[UploadFile] = File(...)):
    '''Reads a numpy file and predcits traffic'''
    # Check if file
    if len(files) > 1:
        raise HTTPException(status_code=400, detail="Upload 1 file at a time")
    file = files[0]
    file_extension = file.filename.split(".")[1]
    if file_extension not in functions.get_possible_extensions():
        raise HTTPException(status_code=400, detail="Filetype not supported")

    # Reading in file and changing it to numpy array
    array_to_process = functions.read_file(await file.read(), file_extension)
    # check_shape_or_raise(array_to_process)

    # Converting numpy error to 2 needed arrays for the model to predict traffic
    try:
        input_conv, input_val = functions.process_numpy(
            array_to_process, file_extension)
    except:
        raise HTTPException(
            status_code=500, detail="An error occured while parsing the file, please try again after checking the file for errors")

    # sending data to model to predict traffic
    try:
        return {"predictions": functions.predict_trafic(input_conv, input_val).tolist()}
    except:
        raise HTTPException(
            status_code=500, detail="An error occured while trying to predict the traffic, please try again after checking the file for errors")

    raise HTTPException(500, 'Unexpected error occured.')

@app.post("/predict/")
async def predict_from_nparray(body: NumpyBody):
    check_shape_or_raise(body.Array)
    try:
        input_conv, input_val = functions.process_numpy(
            body.Array, 'npy')
    except:
        raise HTTPException(500, "An error occured while parsing the file, please try again after checking the file for errors")


    # sending data to model to predict traffic
    try:
        return {"predictions": functions.predict_trafic(input_conv, input_val).tolist()}
    except:
        raise HTTPException(500, "An error occured while trying to predict the traffic, please try again after checking the file for errors")

    raise HTTPException(500, 'Unexpected error occured.')

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')