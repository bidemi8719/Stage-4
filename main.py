# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:39:56 2020

@author: NMSU
"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model 

#def main():
       
#method = sys.argv[1]
#Dataset = sys.argv[2]


# load from the from the saved model
history1 = load_model('CNN_Chile_Disease_vs_Normal_1')
history2 = load_model('CNN_dropout_Chile_Disease_vs_Normal_2')



# for CNN model
acc = history1.history['acc']
val_acc = history1.history['val_acc']
loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# for CNN with Droupout regularized technique
acc = history2.history['acc']
val_acc = history2.history['val_acc']
loss = history2.history['loss']
val_loss = history2.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# for CNN with Data augmentation 






