import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/mophology.png",0)
kernel=np.ones((5,5),np.uint8)

erored=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)


plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(erored),plt.title('Averaging')
plt.xticks([]),plt.yticks([])
plt.show()