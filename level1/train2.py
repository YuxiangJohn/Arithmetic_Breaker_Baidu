
# coding: utf-8

# In[2]:

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import random
import cv2


# In[37]:


number = '0123456789'
sign  ='+-*'
bracket = '()'
characters = number + sign + bracket+' '
mix_nb='0123456789(((((((((('
width,height, n_len,n_class=180, 60, 7, 16


# In[6]:

data = np.genfromtxt('train/labels.txt',dtype='str')
label=data[:,0]


# In[35]:

def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    
    while True:
        for i in range(batch_size):
            file_num = random.randint(0, 90000)
            filename = str(file_num) + '.png'
            X[i] = cv2.imread('train/'+filename)
            
            
            y_str = label[file_num]
            if len(y_str)<7:
                y_str += '  '
            for j, ch in enumerate(y_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# In[42]:

def valid_gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    
    while True:
        for i in range(batch_size):
            file_num = random.randint(90001, 99999)
            filename = str(file_num) + '.png'
            X[i] = cv2.imread('train/'+filename)
            
            y_str = label[file_num]
            if len(y_str)<7:
                y_str += '  '
            for j, ch in enumerate(y_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y


# In[ ]:

from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


# In[ ]:

model = load_model('match.h5')
model.fit_generator(gen(), samples_per_epoch=90000, nb_epoch=25, 
                    nb_worker=2, pickle_safe=True, 
                    validation_data=valid_gen(), nb_val_samples=10000)
model.save('match2.h5')

