# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:35 2020

@author: TEMITAYO
"""
import time
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
import pickle


#===============================================================
pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y= pickle.load(pickle_in)

print(y[0:10])

import keras.backend
print('The Baackend is: ', keras.backend.backend())

print('The variable type of data is:', X.dtype)
print('the shape of X is: ', X.shape)

X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_datagen = ImageDataGenerator(horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.fit(X_train)
validation_generator = test_datagen.fit(X_test)




#----------------- CNN model 
img_size = 50
          
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
input_shape=(img_size, img_size, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary()) 

from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


history1 = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
                   steps_per_epoch=len(X_train) / 32, epochs=30, validation_data = test_datagen.flow(X_test, y_test, batch_size=20),
                   validation_steps=len(X_test) / 20)

# save the model
model.save('CNN_Chile_Disease_vs_Normal_1')

