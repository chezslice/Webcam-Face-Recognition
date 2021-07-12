import os
import cv2

# Identify the classifier.
cascpath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascpath)

# Assign faceCascade onto the webcam.
vid_capture = cv2.VideoCapture(0)

while True:
    # Read the frame by frame.
    ret, frames = vid_capture.read()

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # Draw a border around the face.
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x,y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Closing the program.
vid_capture.release()
cv2.destroyAllWindows()

# Rememeber to test the code or bad thing will happen :).
#print("Test Code")