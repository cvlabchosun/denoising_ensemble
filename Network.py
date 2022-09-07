import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 8,8

from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
#import keras
import numpy as np
import os
from matplotlib.pyplot import figure
#from keras.backend import tensorflow_backend
from tensorflow.keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from skimage.metrics import structural_similarity as ssim
import math
import cv2
import glob
import re
import tensorflow as tf




def r1(input_tensor, features ):
    x = Conv2D(features, 3, activation='relu', padding='same')(input_tensor)
    x = Conv2D(features, 3, padding='same')(x)
    return add([input_tensor, x])


# Multi-Activation Feature Ensemble Module

def actc(x):      
    x1 = Activation('relu')(x)
    x2 = Activation('sigmoid')(x)
    x3 = Lambda(lambda x: x[0]*x[1])([x2,x])
    x4 = Activation('softplus')(x)
    x4=Activation('tanh')(x4)
    x5 = Lambda(lambda x: x[0]*x[1])([x4,x])
    c1= Conv2D(7, kernel_size=3, strides=1, padding='same')(x1)  
    c2= Conv2D(7, kernel_size=5, strides=1, padding='same')(x3)  
    c3= Conv2D(7, kernel_size=7, strides=1, padding='same')(x5)  
    cx=concatenate([c1,c2,c3], axis = 3)
    y= Conv2D(3, kernel_size=3, strides=1, padding='same')(cx)
    return y



# Residual feature Aggregation Module


def mdsr1(ix,f):
    x=Conv2D(f, kernel_size=5, strides=1, padding='same')(ix)
    x=Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
    x=Conv2D(f, kernel_size=1, strides=1, padding='same')(x)
    
    x1=r1(x,f)
    x1=r1(x1,f)
    x2=r1(x,f)
    x2=r1(x2,f)
    x3=r1(x,f)
    x3=r1(x3,f)
    x=add([x1,x2,x3])
    x=concatenate([x,x1,x2,x3 ], axis = 3)
    x=Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    return x 



# Multi-activated Cascaded Aggregation

def rcat(x,f):
    y1= Conv2D(f//4, kernel_size=3, strides=1, padding='same')(x)
   
    x1= Conv2D(16, kernel_size=3, strides=1, padding='same')(x)
    x1= Conv2D(16, kernel_size=1, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=5, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=7, strides=1, padding='same')(x1)
    x1= Conv2D(16, kernel_size=3, strides=1, padding='same')(x1)
    a1=actc(x1)
    a2=actc(y1)
    c1=concatenate([a1,a2])
    c2=  Conv2D(f//4, kernel_size=1, strides=1, padding='same')(c1)
    return add([c2,y1])




# densely residual feature
def rdn(x,f):
    y1 = Conv2D(f, kernel_size=3, strides=1, padding='same')(x)
    y2 = Conv2D(f, kernel_size=3, strides=1, padding='same')(y1)
    a1 = add([y1,y2])
    y3 = Conv2D(f, kernel_size=3, strides=1, padding='same')(a1)
    a2 = add([y3,a1])
    y4 = Conv2D(f, kernel_size=3, strides=1, padding='same')(a2)
    a3 = add([a1,a2,y4])
    c  = concatenate([a1,a2,a3])
    return  Conv2D(f//4, kernel_size=3, strides=1, padding='same')(c)
    

# Proposed model........................................................................

input_img = Input(shape=(None,None,3))
x= Conv2D(64, kernel_size=5, strides=1, padding='same')(input_img)  
x= Conv2D(64, kernel_size=3, strides=1, padding='same')(x)  
x= Conv2D(64, kernel_size=1, strides=1, padding='same')(x)  

f1 = actc(x)
f2 = mdsr1(x,32)
f3 = rdn(x,128)
f4 = rcat(x,16)

inp=concatenate([f1,f2,f3,f4])  
 
x=Conv2D(3, 3, dilation_rate=2, strides=1, padding='same')(inp) 
 
 
model = Model(input_img, x)
model.compile(loss='mse', optimizer='adam',    metrics=[psnr,ssim, "accuracy"])


#.............................................................................................

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.4,
                              patience = 6,
                              verbose = 1,
                              min_delta = 0.0001)

filepath="Bdtrsss1.hdf5"
checkpoint = ModelCheckpoint(filepath, 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min')


opr= keras.optimizers.Adam(lr=0.000003, clipnorm=0.001) 


# your custom loss is a user defiend loss function


model.compile(loss= your custom loss , optimizer='adam',    metrics=[psnr,ssim, "accuracy"])


# custom to the user.............................
#nsa = load noisy data

#imga = load clean data

#...............................................




history = model.fit(nsa,imga,validation_split=0.2 , epochs=7000, batch_size=8, callbacks = [checkpoint,reduce_lr],shuffle=True)



