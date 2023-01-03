import numpy as np 
import cv2 as cv 
import os


path_cascade_classifier = r"D:\GitHub\Face_Detection_and_Recognition\Face Recognition\haarcascade_frontalface_default.xml"
Dir_train = r"D:\GitHub\Face_Detection_and_Recognition\Face Recognition\Resources\Faces\train"

people_list = []

for folder in os.listdir(Dir_train):
    people_list.append(folder)

# Face detection classifier
haar_cascade = cv.CascadeClassifier(path_cascade_classifier)

# features = np.load('features.npy', 'allow_pickle = True')
# labels = np.load('labels.npy')

face_recognizor = cv.face.LBPHFaceRecognizer_create()
face_recognizor.read('face_trained.yml')

img = cv.imread(r"D:\GitHub\Face_Detection_and_Recognition\Face Recognition\Resources\Faces\val\mindy_kaling\1.jpg")

img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', img_gray)

# Drawing rectanle around the detected faces in the image
faces_rect = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 1)
for (x, y, w, h) in faces_rect:
    faces_roi = img_gray[y:y+h, x:x+w]
    labels,confidence = face_recognizor.predict(faces_roi)
    print(f'Label: {people_list[labels]}, with a confidence of {confidence}')
    cv.putText(img, str(people_list[labels]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, 
        (0, 255, 0), thickness = 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)


cv.imshow('Detecte_face', img)

cv.waitKey(0)