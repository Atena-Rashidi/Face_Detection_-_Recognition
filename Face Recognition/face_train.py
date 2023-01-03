import os
import cv2 as cv
import numpy as np 

Dir_train = r"D:\GitHub\Face_Detection_and_Recognition\Face Recognition\Resources\Faces\train"
path_cascade_classifier = r"D:\GitHub\Face_Detection_and_Recognition\Face Recognition\haarcascade_frontalface_default.xml"

# Face detection classifier
haar_cascade = cv.CascadeClassifier(path_cascade_classifier)



people_list = []

for folder in os.listdir(Dir_train):
    people_list.append(folder)

print(people_list)

''' Now, we should make a function that loops over all folder and then all images and detect faces 
and put it in a list. We should have two list, one for feature and the other one should be for 
the image label, meanng that each image belongs to what feature '''

features = []
Labels = []

# loop for each folder/each person in the people list

def creat_train():
    for person in people_list:
        path = os.path.join(Dir_train, person)
        label = people_list.index(person)

        # loop for each image in each person folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            # read img and convert it to grayscale.
            img_array = cv.imread(img_path)
            img_gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor = 1.1, minNeighbors = 3)

            # now we should save this rectangle as the feature of each people with the corresponding label
            for (x, y, w, h) in faces_rect:
                faces_roi = img_gray[y:y+h, x:x+w]
                features.append(faces_roi)
                Labels.append(label)


creat_train()
print(f'Length of feature list: {len(features)}')
print(f'Length of label list: {len(Labels)}')
print('Done! ...')
# We need to convert these features and labels to numpy array.
features = np.array(features, dtype = 'object')
Labels = np.array(Labels)



# We can now use the features and labels that are appended now to train the recognizer on i
# Instantiate the face recognizer that instants from:
face_recognizor = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and the label list
face_recognizor.train(features, Labels)

# We can save the train model and use it in an other program  
# We can save these features and labels
face_recognizor.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', Labels)