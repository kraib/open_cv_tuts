import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/opencvlogo.png",0)


ret,thresh=cv2.threshold(img,127,255,0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(0,255,0),3)


cv2.imshow('image',img)

cv2.waitKey(0)