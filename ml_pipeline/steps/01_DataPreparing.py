"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import json
import os
import sys
import argparse
import traceback
from azureml.core import Run, Experiment, Workspace, Dataset, Datastore

from azureml.core.authentication import AzureCliAuthentication

import pandas as pd
import numpy as np

import itertools
import logging
import math

import cv2

from tensorflow.keras.utils import to_categorical

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

def sincostime(timestamp):
    return [np.cos((timestamp.hour * 60 + timestamp.minute) / 60 / 24 * 2 * 3.141592), np.sin((timestamp.hour * 60 + timestamp.minute) / 60 / 24 * 2 * 3.141592)]

def jam_resample_travel_times(travel_times, segments_size=60, minutes_size=90):
    return cv2.resize(travel_times, dsize=(minutes_size, segments_size))
    
def convert_data(df, train = True):
        X = []
        y = []

        if train:
            df_range = range(round(df.shape[0] * 0.8)) # 80% Train
        else:
            df_range = range(round(df.shape[0] * 0.8), df.shape[0]) # 20% test
        
        for idx, sample in df.iloc[df_range].iterrows():
            xsample = \
            np.asarray([sample.dayOfWeek, *sincostime(sample.timestamp), sample.weekend, *sample.junctions,\
                        *jam_resample_travel_times(np.asarray(sample.actualTravelSpeeds).astype('float64'))[:,:30].flatten(), \
                        *jam_resample_travel_times(np.asarray(sample.optimalTravelSpeeds).astype('float64'))[:,0].flatten(), \
                        sample.roadClass, sum(sample.lengths)/100000])

            ysample = \
            np.asarray([*jam_resample_travel_times(np.asarray(sample.actualTravelSpeeds).astype('float64'))[:,30:].flatten()])
            

            X.append(xsample)
            y.append(ysample)
            if len(y) % 500 == 0:
                logging.debug(f"Already processed {len(y)} samples!")
        
        logging.debug(f"Converting to Numpy arrays!")
        X = np.asarray(X)
        y = np.asarray(y)
                
        return (X, y)

def prepare_data(ws):
    data = os.path.join(os.getcwd(), 'data')
    os.makedirs(data, exist_ok=True)
    dataset = Dataset.get_by_name(ws, name='TrafficDatav3')
    dataset.download(target_path=data, overwrite=True)

    try:
        df = np.load('data/df.npy', allow_pickle=True)
        df = pd.DataFrame(data=df, columns=['eventID', 'roadClass', 'alertCategory', 'timestamp', 'dayOfWeek',
       'weekend', 'segmentIDs', 'junctions', 'optimalTravelSpeeds',
       'optimalTravelTimes', 'lengths', 'actualTravelTimes',
       'actualTravelSpeeds'])
        
        X_train, y_train = convert_data(df, train = True)
        X_test, y_test = convert_data(df, train = False)
        
        X_train_conv = np.reshape(X_train[:,7:1867],(-1,60,31,1)) # a Square representation of 60 segments, 30 minutes of "ActualSpeeds" and 1 minute of "OptimalSpeeds"
        X_train_val = np.concatenate((X_train[:,:7],X_train[:,1867:]),axis = 1) # The rest of the data
        X_test_conv = np.reshape(X_test[:,7:1867],(-1,60,31,1)) # a Square representation of 60 segments, 30 minutes of "ActualSpeeds" and 1 minute of "OptimalSpeeds"
        X_test_val = np.concatenate((X_test[:,:7],X_test[:,1867:]),axis = 1) # The rest of the data
        
        df_test = df.iloc[range(round(df.shape[0] * 0.8), df.shape[0])]
        
        train_test_dir = os.path.join(os.getcwd(), './data/train-test')
        os.makedirs(train_test_dir, exist_ok=True)
        df_test.to_pickle('./data/train-test/test-dataframe.pkl')

        return X_train, X_test, y_train, y_test, X_train_conv, X_train_val, X_test_conv, X_test_val
    
    except Exception as e:
        print(e)
        sys.exit(0)


def upload_data(datastore, target_path, **kwargs):
    train_test_dir = os.path.join(os.getcwd(), 'data/train_test')
    os.makedirs(train_test_dir, exist_ok=True)  
    np.savez('data/train_test/' + file_name, **kwargs)
    datastore.upload(src_dir='data/train_test', target_path='data/' + target_path, overwrite=True)

def register_dataset(ws, datastore, **kwargs):
    dataset = Dataset.File.from_files(
    [
        (datastore, file_name)
    ],
    validate=False)
    dataset = dataset.register(workspace=ws, **kwargs)


def main():
    global file_name
    cli_auth = AzureCliAuthentication()
    from dotenv import load_dotenv
    # For local development, set values in this section
    load_dotenv()
    workspace_name = os.environ.get("WORKSPACE_NAME")
    resource_group = os.environ.get("RESOURCE_GROUP")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")

    dataset_name = os.environ.get("DATASET_NAME")
    dataset_description = os.environ.get("DATASET_DESCRIPTION")

    datastore_folder = os.environ.get("TRAIN_TEST_DATASTORE_FOLDER")

    file_name = "training_testing_data.npz"

    # run_id useful to query previous runs
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth
    )
    datastore = Datastore(ws)

    X_train, X_test, y_train, y_test, X_train_conv, X_train_val, X_test_conv, X_test_val = prepare_data(ws)   

    upload_data(datastore=datastore, target_path=datastore_folder, X_train_conv=X_train_conv, y_train=y_train, X_train_val=X_train_val,X_test_conv=X_test_conv,X_test_val=X_test_val, y_test=y_test)

    register_dataset(ws, datastore, name=dataset_name, description=dataset_description, create_new_version=True)


if __name__ == '__main__':
    main()
