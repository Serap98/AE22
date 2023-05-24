from pathlib import Path
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os

np.random.seed(42)

class AE22:
    def __init__(self, data_path=None):
        """AE22 library extracts 22 features from 88 size time series using autoencoders trained
        on various types of times series taken from UAE & UCR.
        """
        self.__autoencoder_path = Path(os.getcwd()) / Path('ae22/autoencoder/best.tf')
        self.autoencoder = self.__load_autoencoder()
        self.MIN_TIMESERIES_LENGHT = 88
        self.data_path = data_path
        
    def __load_autoencoder(self):
        K.set_learning_phase(0)
        new_model = tf.keras.models.load_model(self.__autoencoder_path)
        embed_layer = [l for l in new_model.layers if l.name == "embedding"][0]
        encoder = K.function(
            inputs=[new_model.layers[0].input], outputs=[embed_layer.output]
        )
        return encoder
    
    def __load_data_from_path(self):
        if self.data_path is not None and os.path.exists(self.data_path):
            _, ext = os.path.splitext(self.data_path)
            ext = ext[1:]
            if not (ext == 'csv' or ext == 'xlsx'):
                raise Exception('Unsupported format data, supported formats: csv and xlsx')
            try:
                if ext == 'csv':
                    data = pd.read_csv(self.data_path)
                elif ext == 'xlsx':
                    data = pd.read_excel(self.data_path)
                return data
            except:
                raise Exception('Unable to load data, please load manually')
            
    
    def __check_input_type(self, data_type):
        if data_type == pd.DataFrame or data_type == list  or data_type == np.ndarray:
            return
        raise Exception(f'Wrong data type provided, expected list, DataFrame, np.array  but got {data_type}')
    
    def __check_timeseries_length(self, data_shape):
        if len(data_shape) == 1 and data_shape[0] >= self.MIN_TIMESERIES_LENGHT:
            return True
        elif(len(data_shape) == 2 and data_shape[1] >= self.MIN_TIMESERIES_LENGHT):
            return True
        raise Exception(f'Data lenght is {data_shape[0]} which is less than expected {self.MIN_TIMESERIES_LENGHT}')
    
    def __windowed_view(self, arr):
        arr = np.asarray(arr)
        window = 88
        overlap = 0
        window_step = window - overlap
        new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
        new_strides = arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:]
        return as_strided(arr, shape=new_shape, strides=new_strides)
    
    def __reshape_dimensions(self, data):
        # differentiate samples from each other
        data = np.array(data)
        data_shape = data.shape
        data_shape_len = len(data_shape)
        if data_shape_len == 2:
            windows_reshaped = data.reshape(data_shape[0], data_shape[1], 1)
        elif data_shape_len == 3:
            windows_reshaped = data.reshape(data_shape[0], data_shape[2], 1)
        else:
            raise Exception('Wrong dimensionality data')
        return windows_reshaped
    
    def transform(self, data):
        """Processes input data using autoencoder and extracts data. 
        
        Extracts 22 features out of each window of 88 size. If single sample contains more than
        one feature, sample is separated into windows and from each window features are extracted.
        In such case, data is returned as follows: (n_samples, n_windows, 22). 

        Args:
            data (pd.DataFrame, np.array): input data used to extract features.

        Returns:
            numpy.ndarray: extracted features from data.
        """
        if data is None:
            data = self.__load_data_from_path()
        data_type = type(data)
        print('Validation start')
        self.__check_input_type(data_type)
        data_np = np.array(data)
        data_shape = data_np.shape
        self.__check_timeseries_length(data_shape)
        data_shape_len = len(data_shape)
        print('Data was validated')
        print('Windows extraction start')
        if data_shape_len == 1:
            windows = np.array(self.__windowed_view(data_np)) #list(map(self.__windowed_view, data_np))
        elif data_shape_len == 2:
            windows = np.array(list(map(self.__windowed_view, data_np)))
        else:
            raise Exception(f'Input dimensions are too high, expected 1D or 2D but got {data_shape_len}')
        print('Windows were formed')
        windows_shape = windows.shape
        print('Feature extraction begins')
        if len(windows_shape) > 2:
            embeddings = []
            n_win = len(windows)
            print(f'In total {n_win} iterations required to extract features')
            for ind, segment in enumerate(windows):
                print(f'Iteration {ind} out of {n_win}')
                segment_reshaped = self.__reshape_dimensions(segment)
                embeds = np.array(self.autoencoder(segment_reshaped)).squeeze()
                embeddings.append(embeds)
        else: 
            windows_reshaped = self.__reshape_dimensions(windows)
            embeddings = np.array(self.autoencoder(windows_reshaped)).squeeze()
        print('Feature extraction completed')
        return np.array(embeddings)
