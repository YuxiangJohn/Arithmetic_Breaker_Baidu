
# coding: utf-8


number = '0123456789'
sign  ='+-*'
bracket = '()'
characters = number + sign + bracket+' '
def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])


import matplotlib.pyplot as plt

import numpy as np
import cv2
from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
result = np.zeros((200000,2), dtype= np.dtype('U25'))
model = load_model('match2.h5')
for i in range(200000):
    img = cv2.imread('validate/'+str(i)+'.png')
    
    pred = model.predict(img.reshape(1,60,180,3))
    str_pred=decode(pred)
    result[i][0]=str_pred.rstrip()
    result[i][1]=str(eval(str_pred))



np.savetxt('result.txt',result, fmt="%s")

