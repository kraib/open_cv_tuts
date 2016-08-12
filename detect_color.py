import cv2

from colorlabeler import ColorLabeler
from shapedetector import ShapeDetector

image = cv2.imread("images/shapes_and_colors")
blurred= cv2.GaussianBlur(image,(5,5),0)
gray=cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
lab=cv2.cvtColor(blurred,cv2.COLOR_BGR2LAB)
thresh=cv2.threshold(gray,60,255,cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[1]

sd = ShapeDetector()
cl = ColorLabeler()

for c in cnts:
    M = cv2.moments(c)
    cX=0
    cY=0
    if(M["m00"]!=0):
        cX=int(M["m10"]/M["m00"])
        cX=int(M["m01"]/M["m00"])
    shape=sd.detect(c)
    color=cl.label(lab,c)
    text="{} {}".format(color,shape)
    cv2.drawContours(image,[c],-1,(0,255,0),2)
    cv2.putText(image,text,(cX,cY),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.5,(255,255,255),2)
cv2.imshow("Image",image)
cv2.waitKey(0);