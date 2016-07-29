import cv2
import numpy as np


def nothing(x):
    pass


image = cv2.imread("images/blue.png")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.namedWindow("image")
cv2.createTrackbar("HL","image",0,179,nothing)
cv2.createTrackbar("SL","image",0,255,nothing)
cv2.createTrackbar("VL","image",0,255,nothing)

cv2.createTrackbar("HU","image",0,179,nothing)
cv2.createTrackbar("SU","image",0,255,nothing)
cv2.createTrackbar("VU","image",0,255,nothing)

switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)
while(1):
    hl = cv2.getTrackbarPos('HL','image')
    sl = cv2.getTrackbarPos('SL','image')
    vl = cv2.getTrackbarPos('VL','image')

    hu = cv2.getTrackbarPos('HU','image')
    su = cv2.getTrackbarPos('SU','image')
    vu = cv2.getTrackbarPos('VU','image')
    # define range of blue color in HSV
    lower_blue = np.array([hl, sl, vl])
    upper_blue = np.array([hu, su, su])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('image',res)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
