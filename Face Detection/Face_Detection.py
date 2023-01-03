
""" Face Detection with Haar Cascades:
It is an Object Detection Algorithm used to identify faces in an image or a real time 
video. The algorithm uses edge or line detection features. 
-------------------------------------------------------------------------------------
source: 
https://github.com/opencv/opencv/tree/master/data/haarcascades 
https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08 

Download: 
haarcascade_frontalface_default.xml, this is the classifier that we use to detect faces!
"""

''' detectMultiScale method: 
returns boundary rectangles for the detected faces (i.e., x, y, w, h). It takes two parameters namely, 
1. scaleFactor
2. minNeighbors 
ScaleFactor determines the factor of increase in window size which initially starts at size “minSize”, 
and after testing all windows of that size, the window is scaled up by the “scaleFactor”, and the window 
size goes up to “maxSize”. '''


import cv2 as cv

path = r"D:\GitHub\Face_Detection_and_Recognition\Face Detection\Pictures\Messi.jpg"
path_cascade_classifier = r"D:\GitHub\Face_Detection_and_Recognition\Face Detection\haarcascade_frontalface_default.xml"

img = cv.imread(path)
cv.imshow('Img', img)

# convert it to grayscale.
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_gray', img_gray)

# Now we need to read from the classifier: 
haar_cascade = cv.CascadeClassifier(path_cascade_classifier)

# Face detection
faces_rec = haar_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 3)
print(f'Number of faces, found = {len(faces_rec)}')

# we can draw a rectangle around the faces were detected.
# Syntax: cv2.rectangle(image, start_point, end_point, color, thickness)

for (x,y,w,h) in faces_rec:
    cv.rectangle(img, (x,y), (x+w,y+h), (0, 255,0), thickness = 2)

cv.imshow("Img_Detected_Face", img)

''' ------------------------------------------------------------ '''
# Now, we want to detect a group of faces:
path_group = r"D:\GitHub\Face_Detection_and_Recognition\Face Detection\Pictures\Argentina_Team.jpg"
img_group = cv.imread(path_group)
cv.imshow('img_group', img_group)

img_group_gray = cv.cvtColor(img_group, cv.COLOR_BGR2GRAY)
faces_rec_group = haar_cascade.detectMultiScale(img_group_gray, scaleFactor = 1.1, minNeighbors = 7)
print(f'Number of faces, found = {len(faces_rec_group)}')

for (x,y,w,h) in faces_rec_group:
    cv.rectangle(img_group, (x,y), (x+w,y+h), (0, 255,0), thickness = 2)

cv.imshow("Img_Detected_Face_Group", img_group)

cv.waitKey(0)
