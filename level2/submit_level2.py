
# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import re
import codecs




number = '0123456789'
chinese = '君不见黄河之水天上来奔流到海不复回烟锁池塘柳深圳铁板烧'
#sign = '='
sign ='+-*/'
bracket = '()'
characters2 = number + sign + chinese + bracket + ' '
width2, height2, n_len2, n_class2=320, 55, 30, len(characters2)




number = '0123456789'
chinese = '君不见黄河之水天上来奔流到海不复回烟锁池塘柳深圳铁板烧'
sign = '='
#sign ='+-*/'
#bracket = '()'
characters= number + sign + chinese + ' '
width1, height1, n_len1, n_class1=180, 75, 8, len(characters)




from keras.models import *
from keras.layers import *
from keras import backend as K




model1 = load_model("base_ctc13.h5") #ctc1
model2 = load_model("base_ctc2_6.h5") #ctc2




import time




start=time.clock()
X_test_1 = np.zeros((1, width1, height1, 3), dtype=np.uint8)
X_test_2 = np.zeros((1, width2, height2, 3), dtype=np.uint8)
file = codecs.open("test1.txt","a","utf-8")
for i in range(0,100000):
    result=""
    X_test_1[0] = cv2.resize(cv2.imread('test/'+str(i)+'_1.png'), (width1, height1), cv2.INTER_LINEAR).transpose(1,0,2)
    y_pred_1 = model1.predict(X_test_1)
    y_pred_1 = y_pred_1[:,2:,:]
    out1 = K.get_value(K.ctc_decode(y_pred_1, input_length=np.ones(y_pred_1.shape[0])*y_pred_1.shape[1], )[0][0])[:, :30]
    out1 = ''.join([characters[x] for x in out1[0]])
    result += out1 +";"
    if os.path.isfile('test/'+str(i)+'_2.png') == True:
        X_test_1[0] = cv2.resize(cv2.imread('test/'+str(i)+'_2.png'), (width1, height1), cv2.INTER_LINEAR).transpose(1,0,2)
        y_pred_1 = model1.predict(X_test_1)
        y_pred_1 = y_pred_1[:,2:,:]
        out1 = K.get_value(K.ctc_decode(y_pred_1, input_length=np.ones(y_pred_1.shape[0])*y_pred_1.shape[1], )[0][0])[:, :30]
        out1 = ''.join([characters[x] for x in out1[0]])   
        result +=  out1 +";"
            
    X_test_2[0] = cv2.resize(cv2.imread('test/'+str(i)+'_0.png'), (width2, height2), cv2.INTER_LINEAR).transpose(1,0,2)
    y_pred_2 = model2.predict(X_test_2)
    y_pred_2 = y_pred_2[:,2:,:]
    out2 = K.get_value(K.ctc_decode(y_pred_2, input_length=np.ones(y_pred_2.shape[0])*y_pred_2.shape[1], )[0][0])[:, :30]
    out2 = ''.join([characters2[x] for x in out2[0]])
    result += out2 + " "+ "0\n"
    
    file.write(result)
    print(i)
    print(result)
    
file.close()
end=time.clock()
print(end-start)


