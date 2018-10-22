
# coding: utf-8



from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import numpy as np
import random




width,height, n_len,n_class=180, 60, 7, 16




imageCap = ImageCaptcha(width=width, height=height)
a = imageCap.generate_image('1-2*3  ')




import string
number = '0123456789'
sign  ='+-*'
bracket = '()'
characters = number + sign + bracket+' '
mix_nb='0123456789(((((((((('




def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            #random_str = ''.join([random.choice(characters) for j in range(4)])
            #gen_str = ''.join(random.choice(mix_nb))
            gen_str=''
            gen_char= random.choice(mix_nb)
            if gen_char == '(':
                gen_str += gen_char              #1 '('
                gen_str += random.choice(number) #2 '0-9'
                gen_str += random.choice(sign)   #3 'sign'
                gen_str += random.choice(number) #4 '0-9'
                gen_str += ')'                   #5 ')'
                gen_str += random.choice(sign)   #6 'sign'
                gen_str += random.choice(number) #7 '0-9'
            else:
                gen_str += gen_char              #1 '0-9'
                gen_str += random.choice(sign)   #2 'sign'
                gen_char = random.choice(mix_nb)
                if gen_char == '(':
                    gen_str += gen_char           #3 '('
                    gen_str += random.choice(number) #4 '0-9' 
                    gen_str += random.choice(sign)   #5 'sign'
                    gen_str += random.choice(number) #6 '0-9' 
                    gen_str += ')'                   #7 ')'
                else:
                    gen_str += gen_char              #3 '0-9'
                    gen_str += random.choice(sign)   #4 'sign'
                    gen_str += random.choice(number) #5 '0-9'
                    gen_str += ' '
                    gen_str += ' '

                
            X[i] = imageCap.generate_image(gen_str)
            
            for j, ch in enumerate(gen_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y



from keras.layers import *
from keras.models import *
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


#VGG16
def train_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()

    
    input = Input(shape=(60,180,3),name = 'image_input')

    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dropout(0.2)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(7)]
   
    model = Model(input=input, output=x)

    
    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



model=train_model()

"""
model.fit_generator(gen(), steps_per_epoch=10000, epochs=5,
                    validation_data=gen(), validation_steps=1000,workers=2)
"""
model.fit_generator(gen(), samples_per_epoch=90000, nb_epoch=25, 
                    nb_worker=2, pickle_safe=True, 
                    validation_data=gen(), nb_val_samples=10000)
model.save('match.h5')




