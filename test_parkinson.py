# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:27:43 2018

@author: harshita
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
import numpy as np
import os
import scipy.io as sio
from scipy.io import wavfile
import librosa
import scipy.signal
from sklearn.utils import shuffle
from keras import layers
from keras.layers import TimeDistributed
from keras.utils import to_categorical
import keras
from sklearn import preprocessing





##########Architecture##############################
model = Sequential()
model.add(Dense(1024, activation='relu',input_shape=(1,26)))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.load_weights('parkinson.h5')
model.summary()
ind=0


##########prediction of new test features#################################
f=open("test_new_features.txt")
lines=f.readlines()
gt_labels=np.zeros(len(lines))
test_labels=np.zeros(len(lines))
for i in range(len(lines)):
    p=lines[i]
    #print(p)
    p=p.strip()
    p=p.split(" ")
    #print(p)
    p=np.array(p)
    p=p.T
    p=p.reshape(1,1,26)
    predict=model.predict(p)
    label=np.argmax(predict)
    test_labels[ind]=label
    gt_labels[ind]=1
    ind=ind+1
    
    
    
############confustion Matrix####################   
conf=np.zeros((2,2),dtype = np.int32)
test_labels = test_labels.astype(int)
gt_labels = gt_labels.astype(int)



for i in range(test_labels.shape[0]):
    print(test_labels[i],gt_labels[i])
    conf[test_labels[i],gt_labels[i]]+=1
    
accuracy = np.sum(np.diag(conf))
accuracy = (float(accuracy)/test_labels.shape[0]) * 100
print(accuracy)