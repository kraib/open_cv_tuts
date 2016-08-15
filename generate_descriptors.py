import cv2
import numpy as np
from os import walk
from os.path import join
import sys

def create_descriptors():
    folder="images/anchors/"
    files=[]
    for(dirpath,dirname,filename) in walk(folder):
        files.extend(filename)
    for f in files:
        print f
        save_descriptors(folder,f,cv2.xfeatures2d.SIFT_create())

def save_descriptors(folder,image_path,feature_detector):
    img = cv2.imread(join(folder,image_path),0)
    keypoints,descriptors=feature_detector.detectAndCompute(img,None)
    descriptor_file = image_path.replace("jpg","npy")
    np.save(join(folder,descriptor_file),descriptors)

create_descriptors()
