# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

##lets load the json data and check what it is
import json
f = open('../input/planesnet/planesnet.json')
planesnet = json.load(f)
f.close()

x=np.array(planesnet['data'])/255.
print('the shape is : ',x.shape)

x=x.reshape([-1,20,20,3])
print(x.shape)

y=np.array(planesnet['labels'])
print('the shape before categorical is :',y.shape)
y=to_categorical(y)
print('the shape is : ',y.shape)


num_classes = 2
resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.25)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_X)

# fits the model on batches with real-time data augmentation:
my_new_model.fit_generator(datagen.flow(train_X, train_y, batch_size=32),
                    steps_per_epoch=3,
                    epochs=2,
                    )