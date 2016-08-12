import helpers
import cv2
import time

image = cv2.imread("images/flowers.png")
(winW, winH) = (128, 128)
for resized in helpers.pyramids(image, 1.5):
    for(x,y,window) in helpers.sliding_window(resized,35,(winW,winH)):
        if window.shape[0] != winH or window.shape[1] !=winW:
            continue
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)