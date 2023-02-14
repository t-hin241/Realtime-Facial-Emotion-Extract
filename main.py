from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Conv2D,Flatten,MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import cv2

#define labels
#emotions = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
#           'Sad': 4, 'Surprise': 5, 'Neutral': 6}

emotions = ['Angry', 'Disgust', 'Fear', 'Happy',
           'Sad', 'Surprise', 'Neutral']
#cv2 display param
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 2

# Load the cascade
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')


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

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
#load pretrained model
model.load_weights("./best_model.h5")

def extract_emo(img):
  img = cv2.resize(img, (48,48))
  img=np.expand_dims(img,axis=0)

  out = model.predict(img)
  return np.argmax(out, axis=-1)

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]
        pred = extract_emo(roi)
        emo = str(emotions[pred[0]])
        out = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        out_n_text = cv2.putText(out, emo, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        #print(emo)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()