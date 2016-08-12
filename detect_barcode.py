from matplotlib import pyplot as plt
import cv2
import numpy as np

image=cv2.imread("images/barcode_01.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
orig=image.copy()

gradX= cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=1,dy=-0,ksize=-1)
gradY= cv2.Sobel(gray,ddepth=cv2.CV_32F,dx=1,dy=-0,ksize=-1)

gradient = cv2.subtract(gradX,gradY)
gradient =cv2.convertScaleAbs(gradient)

blured=cv2.blur(gradient,(9,9))
(_,thresh)=cv2.threshold(blured,255,255,cv2.THRESH_BINARY)

kernel=cv2.getStructuringElement(cv2.MORPH_GRADIENT,(21,7))
closed=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

closed = cv2.erode(closed,None,iterations=4)
closed = cv2.dilate(closed,None,iterations=4)

cnts=cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[1]

cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
rect = cv2.minAreaRect(cnts)
box = np.int0(cv2.BoxPoints(rect))

cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
plt.subplot(121,),plt.imshow(orig,cmap = 'gray'),plt.title('Original')
plt.axis("off")

plt.subplot(122),plt.imshow(image,cmap="gray"),plt.title('Barcodebar')
plt.axis("off")

plt.show()