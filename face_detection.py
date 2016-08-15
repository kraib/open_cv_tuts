import cv2

image=cv2.imread("images/vikings.jpeg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_cascde=cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
faces=face_cascde.detectMultiScale(gray,1.3,5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),((x+w),(y+h)),(255,0,0),2)

cv2.imshow("vinking",image)
cv2.waitKey(0)