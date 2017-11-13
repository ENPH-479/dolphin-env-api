from skimage.measure import block_reduce
from PIL import Image as PImage
import glob
import cv2
import numpy as np


size=300,300

#***need to modify the directory
ima=cv2.imread('/home/lrs/.dolphin-emu/ScreenShots/NABE01/NABE01-7.png')

#ima=cv2.fastNlMeansDenoisingColored(ima,None,20,20,7,21)
ima=cv2.medianBlur(ima,25)
#ima=cv2.GaussianBlur(ima,(20,20),0)

imrs=cv2.resize(ima,size)
cv2.imwrite('/home/lrs/Desktop/new1.png',imrs)
imrg=cv2.cvtColor(imrs,cv2.COLOR_BGR2GRAY)
cv2.imwrite('/home/lrs/Desktop/result.png', imrg)


imrp= cv2.imread('/home/lrs/Desktop/result.png')
data=np.asarray(imrp,dtype='int32')

test=block_reduce(data,block_size=(20,20,3),func=np.max)
cv2.imwrite('/home/lrs/Desktop/test.png', test)
print(test)
qtest=test

qtest[(qtest>0) & (qtest<51)]=21
qtest[(qtest>50) & (qtest<103)]=78
qtest[(qtest>102) & (qtest<155)]=130
qtest[(qtest>154) & (qtest<207)]=182
qtest[(qtest>206) & (qtest<256)]=232

cv2.imwrite('/home/lrs/Desktop/qtest.png', qtest)

