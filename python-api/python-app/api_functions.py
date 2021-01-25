import time
import numpy as np
from io import BytesIO
from numpy import genfromtxt
import pandas as pd
from tensorflow.keras.models import load_model
import json


class ApiFunctions():
    def __init__(self):
        self.version = 0.1
        self.possible_extensions = ['csv', 'npy', 'pkl', 'json']
        self.model = load_model('model/BeMobileModel.h5')

    def get_api_version(self):
        return self.version

    def get_possible_extensions(self):
        return self.possible_extensions

    def read_file(self, file, extension):
        option = self.possible_extensions.index(extension)
        if option == 0:
            csv_np = np.array(genfromtxt(BytesIO(file), delimiter=','))
            return csv_np.reshape(1, csv_np.shape[0])
        if option == 1:
            return np.load(BytesIO(file))
        if option == 2:
            return pd.read_pickle(BytesIO(file)).to_numpy()
        if option == 3:
            return json.load(BytesIO(file))

    def process_numpy(self, nparray, file_ext):
        # CSV
        if file_ext == 'csv':
            input_conv = np.reshape(nparray[7:1867], (-1, 60, 31, 1))
            input_val = np.reshape(np.concatenate(
                (nparray[:7], nparray[1867:])), (1, 9))
            return input_conv, input_val
        elif file_ext == 'json':
            input_conv = np.array(nparray['Conv'])
            input_val = np.array(nparray['Val'])
            return input_conv, input_val
        input_conv = np.reshape(nparray[:, 7:1867], (-1, 60, 31, 1))
        input_val = np.reshape(np.concatenate(
            (nparray[:, :7], nparray[:, 1867:1869]), axis=1), (1, 9))
        return input_conv, input_val

    def predict_trafic(self, input_conv, input_val) -> np.ndarray:
        print("Started timing model prediction")
        start = time.time()
        returnPrediction = self.model.predict([input_conv, input_val])
        stop = time.time()
        print(f"Predicted model, elapsed time: {stop - start}")
        return returnPrediction
