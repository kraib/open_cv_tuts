import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/opencvlogo.png")
# kernel= np.ones((5,5),np.float32)/25
dist = cv2.GaussianBlur(img,(5,5),0)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dist),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.show()