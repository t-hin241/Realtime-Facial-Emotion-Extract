import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import gc
#read data
dataset = pd.read_csv('./dataset/fer20131.csv')

#define labels
emotions = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
           'Sad': 4, 'Surprise': 5, 'Neutral': 6}

#extract labels and images from dataset
images=[]
labels=[]
for i in range(0,len(dataset)):
  if dataset['emotion'][i]==0 or dataset['emotion'][i]==3 or dataset['emotion'][i]==4 or dataset['emotion'][i]==5 or dataset['emotion'][i]==2:
    d=dataset['pixels'][i].split(' ')
    o=np.array(d).astype('float32')
    o=o.reshape((48,48))
    o=cv2.cvtColor(o,cv2.COLOR_GRAY2BGR)
    images.append(o)
    labels.append(int(dataset['emotion'][i]))
    gc.collect()

images=np.array(images)
labels=np.array(labels)
print('tmp label', labels[0])
#save dataset array for training
np.save("./dataset/images01",images)
np.save("./dataset/labels01",labels)