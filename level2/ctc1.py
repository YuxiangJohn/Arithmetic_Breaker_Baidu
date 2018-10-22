
# coding: utf-8


number = '0123456789'
chinese = '君不见黄河之水天上来奔流到海不复回烟锁池塘柳深圳铁板烧'
sign = '='
#bracket = '()'
characters = number + sign + chinese +' '
width, height, n_len, n_class=180, 75, 8, len(characters)


import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import re


from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)





from keras.models import *
from keras.layers import *
rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

x = Dense(32, activation='relu')(x)

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
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')





def gen(batch_size=64):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    
    y = np.zeros((batch_size,n_len), dtype=np.uint8) 
    
    while True:
        for i in range(batch_size):
            file_num = random.randint(0, 99999)
            while file_num == 27840:
                file_num = random.randint(0, 99999)
                
            file_n = random.randint(1,2)
            filename = str(file_num) + '_' + str(file_n) + '.png'
            
            if os.path.isfile('train1/' + filename) == False:
                file_n = 1
                filename = str(file_num) + '_' + str(file_n) + '.png'
                
            X[i] = cv2.resize(cv2.imread('train/'+filename), (width, height), cv2.INTER_LINEAR).transpose(1,0,2)
            #resize
            #label
            label_full = re.split(r'[;,\s]\s*', label[file_num])            
            
            if file_n == 1:
                y_str = label_full[0]
            else:
                y_str = label_full[1]
            
            if len(y_str)<8:
                for i in range(8-len(y_str)):
                    y_str += ' '
            
            y[i] = [characters.find(x) for x in y_str]
                
        yield [X, y,np.ones(batch_size)*int(conv_shape[1]-2),np.ones(batch_size)*n_len], np.ones(batch_size)





def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = gen(128)
    for i in range(batch_num):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_num





from keras.callbacks import *

class Evaluate(Callback):
    def __init__(self):
        self.accs = []
    
    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model)*100
        self.accs.append(acc)
        print
        print 'acc: %f%%'%acc

evaluator = Evaluate()



model.fit_generator(gen(128), samples_per_epoch=90000, nb_epoch=25,
                    callbacks=[EarlyStopping(patience=5), evaluator],
                    validation_data=gen(), nb_val_samples=10000)
model.save('match_ctc1.h5')

