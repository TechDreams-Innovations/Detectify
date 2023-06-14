import numpy as np
import cv2

EyeCascade = cv2.CascadeClassifier('Cascades/EyeCascade.xml')
FaceCascade = cv2.CascadeClassifier('Cascades/FaceCascade.xml')
SmileCascade = cv2.CascadeClassifier('Cascades/SmileCascade.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) 
cap.set(4,480) 
while True:
    ret, img = cap.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FaceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,      
        minSize=(30, 30)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = EyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)      
        smile = SmileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )      
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
        
cap.release()
cv2.destroyAllWindows()
