import cv2
import numpy as np


def is_contour_bad(c):
    peri = cv2.arcLength(c, True)
    approx=cv2.approxPolyDP(c,0.002*peri,True)
    return not len(approx)==4