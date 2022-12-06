#import OpenCV
import cv2

#set classifier as haarcascade_frontal_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture video stream from webcam (value 0)
cap = cv2.VideoCapture(0)

#run image processing to create a gray image then produce detections and draw a rectangle
while True:
    _, img = cap.read()
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #get detections
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    #draw rectangles around detections 
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('img', img)
    
    
    #exit when esc key pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
        
#capture release        
cap.release()
