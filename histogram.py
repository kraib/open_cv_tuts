import cv2
import numpy

image=cv2.imread("images/blue.png",0)
hist=cv2.calcHist([image],[0],None,[256],[0,256])
print hist