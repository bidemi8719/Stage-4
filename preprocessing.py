# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:34 2020

@author: TEMITAYO
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
 

DATAdir = '.../All_chili_data'  #'C:/Users/TEMITAYO/Pictures/dddd'
img_size = 50

CATEGORIES = ["disease", "normal"]

#for category in CATEGORIES :
path = os.path.join(DATAdir) # path to normal and disease
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    new_array = cv2.resize(img_array, (img_size, img_size))
    #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    plt.imshow(new_array)
    plt.show()
    break
#break
    

training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATAdir, category) # path to normal and disease
        class_num =CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
        
       
create_training_data()

print(len(training_data))


import random
random.shuffle(training_data)

x = []
y = []


for features, lable in training_data:
    x.append(features)
    y.append(lable)
    
X = np.array(x).reshape(-1, img_size, img_size, 3)


#saving whAT we have done

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()





