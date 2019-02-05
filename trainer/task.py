
# coding: utf-8

import io
from tensorflow.python.lib.io import file_io
import argparse
import tensorflow as tf
import numpy as np 
import pandas as pd 
from pandas.compat import StringIO
import librosa 
import os 
import soundfile as sf
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import model_from_json
print(tf.__version__) 
print(keras.__version__) 


def initialise_training_set():
    path_train = 'gs://genderclassification/Input/Data/x_train.csv'
    print('downloading x_train.csv file from', path_train)     
    file_stream = file_io.FileIO(path_train, mode='r')
    x_train_data = pd.read_csv(StringIO(file_stream.read()))
    x_train = x_train_data.values 
    return x_train

   

def initialise_test_set():
    path_test = 'gs://genderclassification/Input/Data/x_test.csv'
    print('downloading x_test.csv file from', path_test)     
    file_stream = file_io.FileIO(path_test, mode='r')
    x_test_data = pd.read_csv(StringIO(file_stream.read()))
    x_test = x_test_data.values 
    return x_test

def load_labels():
    path_y_train = 'gs://genderclassification/Input/Data/y_train.csv'
    print('downloading y_train.csv file from', path_y_train)     
    file_stream = file_io.FileIO(path_y_train, mode='r')
    y_train_data = pd.read_csv(StringIO(file_stream.read()))
    y_train = y_train_data.values 
    path_y_test = 'gs://genderclassification/Input/Data/y_test.csv'
    print('downloading y_test.csv file from', path_y_test)     
    file_stream = file_io.FileIO(path_y_test, mode='r')
    y_test_data = pd.read_csv(StringIO(file_stream.read()))
    y_test = y_test_data.values
    print(y_train.shape)
    print(y_test.shape)
    return y_train, y_test


def model(x_train,x_test,y_train,y_test):
    model = Sequential() 
    model.add(LSTM(128, input_shape=(x_train.shape[1:]), kernel_initializer='normal', activation = 'relu'))
    model.add(Dense(64, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(0.30))
    model.add(Dense(16, kernel_initializer='normal', activation = 'relu'))
    model.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))
    return model

def main(job_dir):
    with tf.device('/device:GPU:0'):
        x_train = initialise_training_set()
        x_test = initialise_test_set()
        y_train, y_test = load_labels()
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
        Model = model(x_train,x_test,y_train,y_test)
        Model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy','binary_accuracy'])
        Model.summary()
        Model.fit(x=x_train,y=y_train, epochs=64, batch_size=32)
        Model.save('model.h5')
        with file_io.FileIO('model.h5', mode='r') as input_f:
            with file_io.FileIO(job_dir + 'model/model.h5', mode='w+') as output_f:
                output_f.write(input_f.read())
        score = Model.evaluate(x_test, y_test, batch_size=32)
        print(score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__
    job_dir = arguments.pop('job_dir')
    
    main(job_dir)
