import cv2
import numpy
from matplotlib import pyplot as plt

image=cv2.imread("images/blue.png")

color=('b','g','r')
for i,col in enumerate(color):
    hist=cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.show()
