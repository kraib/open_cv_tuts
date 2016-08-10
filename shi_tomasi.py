import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/chess.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners= cv2.goodFeaturesToTrack(gray,30,0.01,5)
print (corners)
corners=np.int0(corners)
print (corners)

for i in corners:
    print i
    x,y=i.ravel()
    print i.ravel()
    cv2.circle(img,(x,y),3,255,2)
plt.imshow(img),plt.show()
