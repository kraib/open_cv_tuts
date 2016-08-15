import cv2
import numpy as np


def read_images():
    c = 0
    X, y = [], []

    img1 = cv2.imread("faces/2.jpg", 0)
    X.append(np.asarray(img1, dtype=np.uint8))
    y.append(1)
    img3 = cv2.imread("faces/4.jpg", 0)
    X.append(np.asarray(img3, dtype=np.uint8))
    y.append(3)
    img4 = cv2.imread("faces/6.jpg", 0)
    X.append(np.asarray(img4, dtype=np.uint8))
    y.append(4)
    img5 = cv2.imread("faces/8.jpg", 0)
    X.append(np.asarray(img5, dtype=np.uint8))
    y.append(4)
    return [X, y]

[X,y]=read_images()
y=np.asarray(y,dtype=np.int32)

model=cv2.face.createEigenFaceRecognizer()
model.train(np.asarray(X),np.asarray(y))
camera =cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
while(True):
    read,img=camera.read()
    faces=face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        roi=gray[x:x+w,y:y+h]
        roi = cv2.resize(roi, (200, 200),interpolation=cv2.INTER_LINEAR)
        params = model.predict(roi)
        print "Label: %s, Confidence: %.2f" % (params[0], params[1])
        cv2.putText(img, "Kraiba Found", (x, y - 20),cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.rectangle(img,(x,y),((x+w),y+h),(255,255,0),1)
    cv2.imshow("camera", img)
    if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
        break
cv2.destroyAllWindows()