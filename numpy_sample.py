import os

import numpy as np
from matplotlib import pyplot as plt

cwd= os.getcwd()
image_dir =  os.path.join(cwd,"images")
files =[f for f in os.listdir(image_dir) if f.endswith('.jpg')]

redperimage=[]
greenperimage=[]

for image in files:
    img=plt.imread(os.path.join(image_dir.image))
    print img
    reds = img[:,:,0]
    print reds
    redperimage.append(np.sum(reds))
    greens=img[:,:,0]
    greenperimage.append(np.sum(greens))
    print greens
redpermage=np.array(redperimage,dtype=float)
redperimage=np.array(redperimage,dtype=float)
ratio =redperimage/greenperimage

plt.subplot(211)
plt.plot(range(0,len(redperimage)),redperimage,'ro')
plt.plot(range(0,len(greenperimage)),greenperimage,'go')

plt.subplot(212)
plt.plot(range(0,len(ratio)),ratio,'ko')
plt.show()