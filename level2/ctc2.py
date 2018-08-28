
# coding: utf-8

# In[ ]:

number = '0123456789'
chinese = '君不见黄河之水天上来奔流到海不复回烟锁池塘柳深圳铁板烧'
sign = '+-*/'
bracket = '()'
characters = number + sign + chinese + bracket +' '
width, height, n_len, n_class=350, 55, 33, len(characters)


# In[ ]:

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import re
import codecs
label = np.fromiter(codecs.open("labels.txt", encoding="utf-8"), dtype="<U100")


# In[ ]:

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# In[ ]:

from keras.models import *
from keras.layers import *
from keras import regularizers
from keras.applications.vgg16 import VGG16
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
"""# Block 1
model_vgg16_conv = VGG16(include_top=False)
   
x = model_vgg16_conv(x)
"""
for i in range(3):
    x=Conv2D(32,(3,3),activation='relu',kernel_regularizer=regulatizers.l2(1e-5))(x)
    x=Conv2D(32,(3,3),activation='relu',kernel_regularizer=regulatizers.l2(1e-5))(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    


conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(44, activation='relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = merge([gru_1, gru_1b], mode='sum')

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = merge([gru_2, gru_2b], mode='concat')
x = Dropout(0.25)(x)
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')


# In[ ]:

def gen(batch_size=64):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    #length = []
    label_length=np.ones(batch_size)
    
    
    while True:
        for i in range(batch_size):
            file_num = random.randint(0, 99999)
            
            filename = str(file_num) + '_0.png'
            
            
                
            resized = cv2.resize(cv2.imread('train1/'+filename), (width, height), cv2.INTER_LINEAR).transpose(1,0,2)
            X.append(resized)
            #resize
            #label
            label_full = re.split(r'[;,\s]\s*', label[file_num])            
            
            y_str = label_full[-3]
            label_length[i]=len(y_str)
            y=np.zeros((batch_size,len(y_str)),dtype=np.uint8)
            
            
            y[i] = [characters.find(x) for x in y_str]

        
                
        yield [X, y,np.ones(batch_size)*int(conv_shape[1]-2),label_length], np.ones(batch_size)


# In[ ]:

def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = gen(128)
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :33]
        if out.shape[1] == 29:
            batch_acc += ((y_test == out).sum(axis=1) == 33).mean()
    return batch_acc / batch_num


# In[ ]:

from keras.callbacks import *

class Evaluate(Callback):
    def __init__(self):
        self.accs = []
    
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model)*100
        self.accs.append(acc)
        
        print('acc: %f%%'%acc)

evaluator = Evaluate()


# In[ ]:
for loop in range(10):
    model.fit_generator(gen(1), samples_per_epoch=90000, nb_epoch=25,
                    callbacks=[EarlyStopping(patience=10), evaluator],
                    validation_data=gen(1), nb_val_samples=10000)

    model.save('ctc2_f.h5')
    base_model.save('base_ctc2_f.h5')

