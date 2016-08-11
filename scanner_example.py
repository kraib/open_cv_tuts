import cv2
import imutils
from matplotlib import pyplot as plt

from transform import four_point_transform

image = cv2.imread("images/receipt-scanned.jpg")
ratio = image.shape[0] / 500
orig = image.copy()
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx == 4):
        screenCnt = approx
        break

warped = four_point_transform(image, screenCnt.reshape(4, 2))
for point in screenCnt.reshape(4, 2):
    print point
    cv2.circle(image,(point[0],point[1]), 5, (0,0,255), 4 )
plt.subplot(121),plt.imshow(image),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(warped),plt.title('Warped')
plt.xticks([]),plt.yticks([])
plt.show()