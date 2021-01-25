import argparse
import ast
import os
import numpy as np

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from datetime import datetime
import tensorflow as tf
import keras
from keras import initializers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Flatten, Input, concatenate

from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
import keras.backend as K

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from azureml.core import Run

def conv_model(data, **params):
    '''
    parameters:
    conv_layers: array with number of convolutions (length is number of layers)
    maxpooling: 1 for maxpooling, 0 for AveragePooling2D 
    poolsize: array with poolsizes (length is number of layers)
    conv_dense: array with number of dense elements after the conv part (length is number of layers)
    val_dense: array with number of dense elements for the metadata part (length is number of layers)
    comb_dense: array with number of dense elements for the combined part (length is number of layers)
    dropout: dropout parameter used equally everywhere (only with dense)
    epochs: epochs to train
    '''
    (X_train_conv, X_train_val, y_train, X_test_conv, X_train_val, y_test) = data
    # define two sets of inputs
    
    X_conv = Input(shape=(60,31,1,))
    X_data = Input(shape=(9,))
    
    f=15
    chanDim = -1
    
    dropout = params['dropout']
    
    #### X
    conv_size = params['conv_layers'][0]
    pool_size = params['pool_size'][0]
    
    x = Conv2D(f, conv_size, padding="same")(X_conv)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=pool_size)(x) if params['maxpooling'] \
    else AveragePooling2D(pool_size=pool_size)(x)

    for i in range(1, len(params['conv_layers'])):
        conv_size = params['conv_layers'][i]
        pool = params['pool_size'][i]

        x = Conv2D(f, conv_size, padding="same")(X_conv)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=pool_size)(x) if params['maxpooling'] \
        else AveragePooling2D(pool_size=pool_size)(x)
    
    x = Flatten()(x)
    for i in range(len(params['conv_dense'])):
        dense_size = params['conv_dense'][i]
        
        x = Dense(dense_size)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        x = Dropout(dropout)(x)
        
    x = Model(inputs=X_conv, outputs=x)

    #### Y
    dense_size = params['val_dense'][0]
    y = Dense(dense_size, activation="relu")(X_data)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    
    for i in range(1, len(params['val_dense'])):
        dense_size = params['val_dense'][i]
        
        y = Dense(dense_size)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Dropout(dropout)(y)

    y = Model(inputs=X_data, outputs=y)

    #### Z
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    
    dense_size = params['comb_dense'][0]
    z = Dense(dense_size, activation="relu")(combined)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    for i in range(len(params['comb_dense'])-1):
        densesize = params['comb_dense'][i+1]
        
        z = Dense(dense_size)(z)
        z = BatchNormalization()(z)
        z = Activation("relu")(z)
        z = Dropout(dropout)(z)
    

    z = Dense(3600, activation="linear")(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001),loss='mse' ,metrics=["accuracy", "MeanSquaredError"])    
    
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=str, dest="batch_size", help="Size of the batch")
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--conv_layers', type=str, dest='conv_layers', help='array with number of convolutions (length is number of layers)')
parser.add_argument('--maxpooling', type=str, dest='maxpooling', help='1 for maxpooling, 0 for AveragePooling2D')
parser.add_argument('--pool_size', type=str, dest='pool_size', help='array with poolsizes (length is number of layers)')
parser.add_argument('--conv_dense', type=str, dest='conv_dense', help='array with number of dense elements after the conv part (length is number of layers)')
parser.add_argument('--val_dense', type=str, dest='val_dense', help='array with number of dense elements for the metadata part (length is number of layers)')
parser.add_argument('--comb_dense', type=str, dest='comb_dense', help='array with number of dense elements for the combined part (length is number of layers)')
parser.add_argument('--dropout', type=str, dest='dropout', help='dropout parameter used equally everywhere (only with dense)')
parser.add_argument('--epochs', type=str, dest='epochs', help='epochs to train')
parser.add_argument('--model_name', type=str, dest='model_name', help='Name of the model')
parser.add_argument('--model_file', type=str, dest='model_file', help='File extension of model')
args = parser.parse_args()

params = {'conv_layers': [(10, 10), (10, 10), (5, 5)],
'maxpooling': False,
'pool_size': [(2, 2), (2, 2), (3, 3)],
'conv_dense': [1200, 800],
'val_dense': [10, 8],
'comb_dense': [1500],
'dropout': 0.15,
'epochs': 20}

print(args.conv_layers)
conv_layers = args.conv_layers.replace('\\', '')
maxpooling = args.maxpooling.replace('\\', '')
pool_size = args.pool_size.replace('\\', '')
conv_dense = args.conv_dense.replace('\\', '')
val_dense = args.val_dense.replace('\\', '')
comb_dense = args.comb_dense.replace('\\', '')
dropout = args.dropout.replace('\\', '')
epochs = args.epochs.replace('\\', '')
batch_size = args.batch_size.replace('\\', '')
print(conv_layers)
params["conv_layers"] = [ast.literal_eval(elem) for elem in conv_layers.split(';')]
params["maxpooling"] = bool(maxpooling)
params["pool_size"] = [ast.literal_eval(elem) for elem in pool_size.split(';')]
params["conv_dense"] = [int(e) for e in conv_dense.split(',')]
params["val_dense"] = [int(e) for e in val_dense.split(',')]
params["comb_dense"] = [int(e) for e in comb_dense]
params["dropout"] = float(dropout)
params["epochs"] = int(epochs)
params["batch_size"] = int(batch_size)

data_folder = args.data_folder
#print('Data folder:', data_folder)

np_arrays = np.load(data_folder)
X_train_conv = np_arrays['X_train_conv']
X_train_val = np_arrays['X_train_val']
y_train = np_arrays['y_train']
X_test_conv = np_arrays['X_test_conv']
X_test_val = np_arrays['X_test_val']
y_test = np_arrays['y_test']

data = [X_train_conv, X_train_val, y_train, X_test_conv, X_test_val, y_test]

# get hold of the current run
run = Run.get_context()

#conv_layers = [args.conv_layers]
#maxpooling = int(args.maxpooling)
#pool_size = [args.pool_size]
#conv_dense = [args.conv_dense]
#val_dense = [args.val_dense]
#comb_dense = [args.comb_dense]
#dropout = float(args.dropout)
#epochs = int(args.epochs)
 

model = conv_model(data, **params)

history_callback = model.fit(
    x=[X_train_conv, X_train_val],
    y=y_train,
    validation_data = ([X_test_conv, X_test_val], y_test),
    epochs = params["epochs"],
    batch_size = params["batch_size"],
    verbose = 1)

run.log("history", history_callback.history)
run.log_list("loss", history_callback.history["loss"])
run.log_list("val_loss", history_callback.history["val_loss"])
run.log_list("val_accuracy", history_callback.history["val_accuracy"])
run.log_list("accuracy", history_callback.history["accuracy"])
run.log_list("mean_squared_error", history_callback.history["mean_squared_error"])
run.log_list("val_mean_squared_error", history_callback.history["val_mean_squared_error"])

#Testing the testset


run.log("batch_size", params["batch_size"])
#run.log("Conv_layers", params["conv_layers"])
run.log("model_name", args.model_name)
run.log("model_file", args.model_file)
#run.log("maxpooling", params["maxpooling"])
#run.log("pool_size", params["pool_size"])
#run.log("conv_dense", params["conv_dense"])
#run.log("val_dense", params["val_dense"])
#run.log("comb_dense", params["comb_dense"])
run.log("dropout", params["dropout"])
run.log("epochs", params["epochs"])



#run.log("Conv_layers: ", conv_layers)
#run.log("Maxpooling: ", maxpooling)
#run.log("Pool_size: ", pool_size)
#run.log("Conv_dense: ", conv_dense)
#run.log("Val_dense: ", val_dense)
#run.log("Comb_dense: ", comb_dense)
#run.log("Dropout: ", dropout)
#run.log("Epochs: ", epochs)

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
model.save(f"outputs/{args.model_file}")
