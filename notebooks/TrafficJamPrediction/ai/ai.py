import os
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
from keras.models import load_model

from ..plotting import Plotting
from ..preprocessing import jam_resample_travel_times
from ..utils import dataframe_timestamp_to_datetime, sincostime


class AI(object):

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.latest_predictions = []
        self.history_callback = []
        self.models = []

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    def prepare_dataframe(self):

        dataframe_timestamp_to_datetime(self.dataframe, 'timestamp', offset=7200, unit='s')

        if len(self.dataframe.iloc[0]['actualTravelTimes'].shape) < 2:
            self.dataframe['actualTravelTimes'] = self.dataframe.apply(lambda row: np.stack(row['actualTravelTimes']), axis=1)

        if len(self.dataframe.iloc[0]['actualTravelSpeeds'].shape) < 2:
            self.dataframe['actualTravelSpeeds'] = self.dataframe.apply(lambda row: np.stack(row['actualTravelSpeeds']), axis=1)

    def prepare_data(self, train = True):
        X = []
        y = []
        df = self.dataframe

        if train:
            df_range = range(round(df.shape[0] * 0.8)) # 80% Train
        else:
            df_range = range(round(df.shape[0] * 0.8), df.shape[0]) # 20% test
        
        for idx, sample in df.iloc[df_range].iterrows():
            xsample = \
            np.asarray([sample.dayOfWeek, *sincostime(sample.timestamp), sample.weekend, *sample.junctions,\
                        *jam_resample_travel_times(np.asarray(sample.actualTravelSpeeds).astype('float32'))[:,:30].flatten(), \
                        *jam_resample_travel_times(np.asarray(sample.optimalTravelSpeeds).astype('float32'), minutes_size=1).flatten(), \
                        sample.roadClass, sum(sample.lengths)/100000])

            ysample = \
            np.asarray([*jam_resample_travel_times(np.asarray(sample.actualTravelSpeeds).astype('float32'))[:,30:].flatten()])
            

            X.append(xsample)
            y.append(ysample)

        X = np.asarray(X)
        y = np.asarray(y)

        if train:
            self.X_train = X
            self.y_train = y
        else:
            self.X_test = X
            self.y_test = y
                
        return (X, y)

    def prepare_convolutional_data(self, X, train = False):
        X_conv = np.reshape(X[:,7:1867], (-1,60,31,1))
        X_val = np.concatenate((X[:,:7], X[:,1867:]), axis = 1)
        
        if train:
            self.X_train_conv = X_conv
            self.X_train_val = X_val
        else:
            self.X_test_conv = X_conv
            self.X_test_val = X_val


        return (X_conv, X_val)

    def save_data(self, directory):
        time_to_save = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        path = Path(f"{directory}/{time_to_save}")
        path.mkdir(parents=True, exist_ok=True)
        np.save(f"{path}/X_train.npy", self.X_train)
        np.save(f"{path}/y_train.npy", self.y_train)
        np.save(f"{path}/X_test.npy", self.X_test)
        np.save(f"{path}/y_test.npy", self.y_test)
        return path

    def load_data(self, directory, data='ALL', absolute = False):
        directory = directory if absolute else sorted(glob(f"{directory}/*/"))[-1]
        data_glob = '*_test' if data == 'TEST' else '*_train' if data == 'TRAIN' else '*'
        filenames = sorted(glob(f"{directory}/{data_glob}.npy")) # sorting means that the order will be preserved.
        if data == 'TEST':
            self.X_test, self.y_test = [np.load(filename) for filename in filenames]
        elif data == 'TRAIN':
            self.X_train, self.y_train = [np.load(filename) for filename in filenames]
        else:
            self.X_test, self.X_train, self.y_test, self.y_train = [np.load(filename) for filename in filenames]
        return (self.X_test, self.X_train, self.y_test, self.y_train)

    def model_train(self, data = None, val_data = None, model = None, *args, **kwargs):
        """
            **kwargs:
                - epochs
                - batch_size
                - verbose
        """
        if self.model is None and model is None:
            raise ValueError("The model is not given")
        elif self.model is not None and model is None:
            model = self.model
        elif self.model is None and model is not None:
            model = model
        else:
            model = model
            # raise Exception("Could not retrieve model to train")

        X_train, y_train = data if data is not None else (self.X_train, self.y_train)
        X_val, y_val = val_data if val_data is not None else (self.X_test, self.y_test)

        self.history_callback.append(model.fit(X_train, y_train, validation_data=(X_val, y_val), **kwargs))
        return self.history_callback[-1]

    def model_predict(self, values, model = None):
        if self.model is None and model is None:
            raise ValueError("The model is not given")
        elif self.model is not None and model is None:
            model = self.model
        elif self.model is None and model is not None:
            model = model
        else:
            raise Exception("Could not retrieve model to predict")

        predictions = model.predict(values)
        self.latest_predictions = predictions

        return predictions

    def model_history(self, history_index = -1):
        plotting = Plotting()

        epochs = self.history_callback[history_index].epoch
        history = self.history_callback[history_index].history
        for key, value in history.items():
            plotting.plotly(x=epochs, y=np.asarray(value), name=key, trace_type="Scatter", mode='lines+markers')
        plotting.figure.update_layout()
        plotting.show()

    def model_save(self, directory):
        time_to_save = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        path = Path(f"{directory}/{time_to_save}")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        return path

    def model_load(self, directory, absolute = False):
        last_model = directory if absolute else sorted(glob(f"{directory}/*/"))[-1]
        self.model = load_model(last_model)
        return self.model

    def dataframe_get_subset(self, subset_size = 1000, full_dataset = True):
        """
            Return a subset of the data (useful for plotting).
            args:
                - subset_size: int
                - full_dataset: bool - If full_dataset is true, the returned dataset is a subset of the full dataframe. Else, the subset comes from an already smaller dataset, the same size of X_test.
            returns:
                - (dataset, test_set)
                    dataset: pd.DataFrame() -- A slice of the original dataframe used to create the X_test data
                    test_set: np.array() -- A slice of the X_test data
        """
        dataframe = self.dataframe.iloc[:subset_size].reset_index(drop=True)
        if full_dataset:
            dataframe = self.dataframe.iloc[round(self.dataframe.shape[0] * 0.8):round(self.dataframe.shape[0] * 0.8) + subset_size ].reset_index(drop=True)
        test_set = self.X_test[:subset_size]
        return (dataframe, test_set)
