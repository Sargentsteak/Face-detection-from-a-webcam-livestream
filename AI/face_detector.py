import cv2
from random import randrange

trained_face_data= cv2.CascadeClassifier('C:\AI\haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture()

#To iterate forever over frames
while True:
     
    #Reads the current frame
    successful_frame_read, frame = webcam.read()

    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
 
    #Draw rectangles around the faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y),(x+w,y+h),randrange(256),randrange(256),randrange(256))

    cv2.imshow('Clever Programmer face detector',frame)
    key = cv2.waitkey(1)

    if key==81 or key==113:
        break


webcam.release()
print("Code Completed")
