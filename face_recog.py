import cv2
import numpy as np
import os 
import csv

recognizer = cv2.face.LBPHFaceRecognizer_create()
yml_path='trainer.yml'
recognizer.read(yml_path)
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
faceDict= {}
with open('sign_up_details.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                faceDict[int(row[5])]=row[2]
faceDict[0]="unknown"

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        
 # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 60):
            name=faceDict[int(id)]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print("Access Denied!!!")
        
        cv2.putText(img, name, (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  

        if id !="unknown":
            print("Access Granted!!!")
        
       
            
    
    cv2.imshow('camera',img)


    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()