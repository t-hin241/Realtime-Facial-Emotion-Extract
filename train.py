import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Conv2D,Flatten,MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


#load preprocessed data
features = np.load("./dataset/images.npy")
labels = np.load("./dataset/labels.npy")
labels=to_categorical(labels)
#train/test split
Xtrain,Xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.2,random_state=7)

#define CNN model
model=Sequential()
model.add(Conv2D(64,kernel_size=(5,5),input_shape=(48,48,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,kernel_size=(5,5),activation='relu'))
model.add(Conv2D(128,kernel_size=(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256,kernel_size=(5,5),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='softmax'))

#model.summary()
#print(model.layers)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#training
model.fit(Xtrain,ytrain,batch_size=64, 
            epochs=20, 
            validation_data=(Xtest,ytest),
            shuffle=True,
)

#save model
model.save("./best_model")
     