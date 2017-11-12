from skimage.measure import block_reduce
from PIL import Image as PImage
import glob
import cv2
import numpy as np
#import matplotlib.

size=300,300
'''
ima=cv2.imread('/home/lrs/.dolphin-emu/ScreenShots/NABE01/NABE01-7.png')
ima= PImage.open( '/home/lrs/.dolphin-emu/ScreenShots/NABE01/NABE01-7.png')
imre=ima.resize(size, PImage.ANTIALIAS)
imre.save('/home/lrs/Desktop/new1.png','PNG')
imrb=imre.convert('1')
imrb.save('/home/lrs/Desktop/result.png')

imrb.load()
data=np.asarray(ima,dtype='int32')
print (data)


block_reduce(ima,block_size=(3,3,1),func=np.max)
'''



#ima=cv2.imread('/home/lrs/.dolphin-emu/ScreenShots/NABE01/NABE01-15.png')
ima=cv2.imread('/home/lrs/Desktop/i.png')

#ima=cv2.fastNlMeansDenoisingColored(ima,None,20,20,7,21)
ima=cv2.medianBlur(ima,25)
#print(ima)
#ima=cv2.GaussianBlur(ima,(20,20),0)
imrs=cv2.resize(ima,size)
cv2.imwrite('/home/lrs/Desktop/new1.png',imrs)
imrg=cv2.cvtColor(imrs,cv2.COLOR_BGR2GRAY)
cv2.imwrite('/home/lrs/Desktop/result.png', imrg)


imrp= cv2.imread('/home/lrs/Desktop/result.png')
data=np.asarray(imrp,dtype='int32')
#print (data)

#temp=np.delete(data,3,axis=2)
#print(temp)
test=block_reduce(data,block_size=(20,20,3),func=np.max)
print(test.size)
cv2.imwrite('/home/lrs/Desktop/test.png', test)




'''
ima=cv2.imread('/home/lrs/.dolphin-emu/ScreenShots/NABE01/NABE01-7.png')
imrss=cv2.resize(ima,(400,400))
cv2.imwrite('/home/lrs/Desktop/new1.png',imrss)


data=np.asarray(imrss,dtype='int32')
#print (data)

#temp=np.delete(data,3,axis=2)
#print(temp)
test=block_reduce(data,block_size=(40,40,3),func=np.max)
cv2.imwrite('/home/lrs/Desktop/grss.png', test)
#print(test)
#imrg=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
#cv2.imwrite('/home/lrs/Desktop/result.png', imrg)
#print(imrg)

imrs=cv2.resize(test,(10,10))
print(imrs)
cv2.imwrite('/home/lrs/Desktop/new2.png',imrs)
'''

