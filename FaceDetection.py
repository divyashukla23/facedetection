import  cv2 as cv
import numpy as np

'''
Mechanism:
Firstly, a classifier(cascade of boosted classifiers working with haar-like features) is trained with a few hundred sample views of a particular object, called positive  examples of same size
that are scaled to same size, and negative examples - arbitrary images of same size 

OpenCV comes with a trainer as well as a detector
'''

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
'''
Parameters of CascadeClassifiers.detectMultiScale
image, matrix of type CV_8U containing the image
objects, vector of rectangles containing the deteccted object
scaleFactor, how much image size is reduced at each image scale
minNeighbours, how many neighbours each candidate rectangle should have to retain it
'''

# # Take the input of the image
# img = cv.imread('I2.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

# For video capture

cap = cv.VideoCapture(0)
while cap.isOpened():
    _,img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 3)

        
    
    # Display the O/P
    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF ==  ord('q'):
        break
    


# Display the image
# cv.imshow('Me', img)
# cv.waitKey(0)
cap.release()
