import os
import random

import numpy as np

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Convolution2D, Activation


# 乱数設定
def set_randvalue(value):
    # Set a seed value
    seed_value= value 
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

def base_model():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1), name='conv1'))
    model.add(layers.MaxPooling2D((2, 2), name='maxpool1'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same", name='conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='maxpool2'))
    model.add(Flatten(name='flatten1')) 

    model.add(Dense(1024, activation='relu', name='dense1'))
    model.add(layers.Dense(10, activation='softmax', name='dense2'))

    return model