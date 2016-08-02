import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("images/tsukuba_1.png",0)
gequ=cv2.equalizeHist(img)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

plt.subplot(2,2,1),plt.imshow(img,cmap='gray'),plt.title('Input'),plt.yticks([]),plt.xticks([])
plt.subplot(2,2,2),plt.imshow(gequ,cmap='gray'),plt.title('Global Equalization'),plt.yticks([]),plt.xticks([])
plt.subplot(2,2,3),plt.imshow(cl1,cmap='gray'),plt.title('Adaptive Histogram Equalization '),plt.yticks([]),plt.xticks([])
plt.show()