from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, Conv1D
from keras.layers.normalization import BatchNormalization
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
model.summary()


feature=[]
label=[]
feature_test=[]
label_test=[]
num=0
#################Training Data####################################
f=open("/home/birds/wd/DCASE/train_parkinson.txt")
lines=f.readlines()
for i in range(len(lines)):
    p=lines[i]
    p=p.strip()
    q=p.split(',')
    data=q[1:27]
    final = preprocessing.scale(data)
    print(len(final),num)
    num=num+1
    #data1=data-np.min(data)
    #inal=data1/float(np.max(data1))
    feature.append(final)
    label.append(q[-1])


####################Validation Data###############################
f=open("/home/birds/wd/DCASE/test_parkinson.txt")
lines=f.readlines()
for i in range(len(lines)):
    p=lines[i]
    p=p.strip()
    q=p.split(',')
    data=q[1:27]
    final = preprocessing.scale(data)
    #data1=data-np.min(data)
    #inal=data1/float(np.max(data1))
    feature_test.append(final)
    label_test.append(q[-1])
    
feature=np.array(feature)
feature=feature.reshape(-1,1,26)
feature_test=np.array(feature_test)
feature_test=feature_test.reshape(-1,1,26)
label=np.array(label)
label=label.reshape(-1,1)
label_test=np.array(label_test)
label_test=label_test.reshape(-1,1)
print(label.shape)
label = to_categorical(label,2) 
label = label.reshape(-1,1,2)
print(label.shape)
print(label_test.shape)
label_test = to_categorical(label_test, 2)
label_test = label_test.reshape(-1,1,2)
print(feature.shape)
print(feature_test.shape)

adada = Adam(lr=0.001,decay = 1e-6)
#sgd = optimizers.SGD(lr=0.1, decay=1e-12, momentum=0.9, nesterov=True)

x_train=feature
y_train=label

x_test=feature_test
y_test=label_test
# compile model
model.compile(loss='categorical_crossentropy', optimizer=adada, metrics=['accuracy'])
checkpoint = ModelCheckpoint('parkinson.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=4, nb_epoch=5,callbacks=callbacks_list, verbose=1)

print("model saved")

    
    