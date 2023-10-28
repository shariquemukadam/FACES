import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\nEnter user id and press <return> ==>  ')

print("\n[INFO] Initializing face capture. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if count < 500:
            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
            count += 1
            cv2.waitKey(100)  # Optional delay to slow down image capture
        else:
            print("\n[INFO] Captured 200 photos. Exiting Program and cleaning up.")
            break

    cv2.imshow('Face Capture', img)

    key = cv2.waitKey(1)
    if key == ord('q') or count == 500:
        break

# Do a bit of cleanup
print("\n[INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()
