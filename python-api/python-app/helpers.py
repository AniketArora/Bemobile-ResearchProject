from fastapi import HTTPException

def check_shape_or_raise(array_to_process):
    if array_to_process.shape != (1, 1869):
        raise HTTPException(
            status_code=400, detail=f"Please make sure the data is shaped like this : (1, 1869) Data shape is {array_to_process.shape}")

    return True
