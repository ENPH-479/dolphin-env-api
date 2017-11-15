from skimage.measure import block_reduce
from PIL import Image as PImage
import glob
import cv2
import numpy as np
import os

size=300,300
sdir='/home/lrs/.dolphin-emu/ScreenShots/NABE01'
count=0
curr_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(curr_path, "../Image")
os.makedirs(file_path, exist_ok=True)

for root,dirs,filenames in os.walk(sdir):
    for f in filenames:
        count+=1
        ima=cv2.imread(os.path.join(sdir,f))

        # Filtering
        #ima=cv2.fastNlMeansDenoisingColored(ima,None,20,20,7,21)
        ima=cv2.medianBlur(ima,25)
        #ima=cv2.GaussianBlur(ima,(20,20),0)

        # Resizing
        imrs=cv2.resize(ima,size)

        # Grayscaling
        imrg=cv2.cvtColor(imrs,cv2.COLOR_BGR2GRAY)

        # Maxpooling
        data=np.asarray(imrg,dtype='int32')
        test=block_reduce(data,block_size=(30,30),func=np.max)


        # Quantization
        qtest=test
        qtest[(qtest>0) & (qtest<51)]=21
        qtest[(qtest>50) & (qtest<103)]=78
        qtest[(qtest>102) & (qtest<155)]=130
        qtest[(qtest>154) & (qtest<207)]=182
        qtest[(qtest>206) & (qtest<256)]=232

        cv2.imwrite(os.path.join(file_path,'%d.png'%count), qtest)

