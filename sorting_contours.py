import numpy as np
import cv2
from matplotlib import pyplot as plt


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-top":
        return True
    if method == "top-to-bottom" or method == "bottom-top":
        i = 1

    boudingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boudingBoxes) = zip(*sorted(zip(cnts, boudingBoxes), key=lambda b: b[1][i], reverse=reverse))
    return (cnts, boudingBoxes)


def draw_contour(image, c, i):
    M = cv2.moments(c)
    cX = 0
    cY = 0
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return image

image = cv2.imread("images/sorted_contours_edge_map.jpg")
accumEdged = np.zeros(image.shape[:2],dtype="uint8")

for chan in cv2.split(image):
    chan=cv2.medianBlur(chan,11)
    edged=cv2.Canny(chan,50,200)
    accumEdged= cv2.bitwise_or(accumEdged,edged)
cnts=cv2.findContours(accumEdged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[1]

cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]
orig=image.copy()

for (i,c) in enumerate(cnts):
    orig=draw_contour(orig,c,i)

(cnts,boundinBoxes)=sort_contours(cnts)

for (i,c) in enumerate(cnts):
    image=draw_contour(image,c,i)


plt.subplot(121,),plt.imshow(orig,cmap = 'gray'),plt.title('Original')
plt.axis("off")

plt.subplot(122),plt.imshow(image,cmap="gray"),plt.title('Sorted')
plt.axis("off")

plt.show()