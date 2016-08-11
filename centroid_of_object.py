import cv2
import imutils
from matplotlib import pyplot as plt
image=cv2.imread("images/shapes_and_colors.jpg")
original=image.copy()
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred=cv2.GaussianBlur(gray,(5,5),0)
thresh=cv2.threshold(blurred,60,555,cv2.THRESH_BINARY)[1]
cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]


for c in cnts:
    M=cv2.moments(c)
    cX=0
    cY=0
    if M["m00"]!=0:
        cX=int(M["m10"] / M["m00"])
        cY=int(M["m01"] / M["m00"])
    cv2.drawContours(image,[c],-1,(0,255,0),2)
    cv2.circle(image,(cX,cY),7,(255,255,255),2)
    cv2.putText(image, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

plt.subplot(2,2,1),plt.imshow(original,cmap = 'gray'),plt.title('Original')
plt.axis("off")

plt.subplot(2,2,2),plt.imshow(thresh,cmap="gray"),plt.title('Threshold')
plt.axis("off")

plt.subplot(2,2,3),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),plt.title('Result')
plt.axis("off")
plt.show()