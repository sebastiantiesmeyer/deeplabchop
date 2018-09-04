#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:46:12 2018

@author: sebastian
"""



from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Conv2DTranspose
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import Sequence

import numpy as np
import os
import pickle
import imageio

input_shape = (715,751)
output_shape = (368,384)



class DeepLab():
    def __init__(self,n_categories):
        self.model = self.build(n_categories)
        
    def build(self, n_outputs):
        model = applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape = (None, None , 3))
        
        for i in range(len(model.layers)//2):
            model.layers[i].trainable=False
        
        x = model.layers[130].output
        
        x = Conv2DTranspose(20, kernel_size=(3,3), strides = (2,2), activation = 'selu')(x)
        x = Conv2DTranspose(10, kernel_size=(3,3), strides = (2,2), activation = 'selu')(x)
        x = Conv2DTranspose(10, kernel_size=(1,1), strides = (1,1), activation = 'selu')(x)        
        x = Conv2DTranspose(10, kernel_size=(3,3), strides = (2,2), activation = 'selu')(x)
        x = Conv2DTranspose(8, kernel_size=(2,2), strides = (1,1), activation = 'selu')(x)
        x = Conv2DTranspose(n_outputs, kernel_size=(1,1),activation = 'sigmoid', activity_regularizer='l2')(x)
                
        model_final = Model(input = model.input, output = x)
        
        model_final.compile(loss = "categorical_crossentropy", 
                            optimizer = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.5), 
                            metrics=["accuracy"])
        
        return model_final



class DeepLabGen(Sequence,ImageDataGenerator):
    def __init__ (self,images, targets, 
                    batch_size, keys=None,
                    rescale = 1./255,
                    horizontal_flip = True,
                    fill_mode = "nearest",
                    zoom_range = 0.05,
                    width_shift_range = 0,
                    height_shift_range=0,
                    rotation_range=10):
    
        ImageDataGenerator.__init__(self, 
                rescale = rescale,
                horizontal_flip = horizontal_flip,
                fill_mode = fill_mode,
                zoom_range = zoom_range,
                width_shift_range = width_shift_range,
                height_shift_range=height_shift_range,
                rotation_range=rotation_range)
        
        self.batch_size = batch_size
        
        if keys is None:
            self.keys = list(images.keys())
        else:
            self.keys=keys
        self.n_samples = len(self.keys)
        self.n = 0
        
        self.images=images
        self.targets={}
        self.fract = np.divide(input_shape,output_shape)
        for i,k in enumerate(images.keys()):
            self.targets[k] = np.zeros(output_shape+(len(targets[k]),))
            for j,l in enumerate(targets[k].keys()):
                for m in range(len(targets[k][l][0])):
                    val_x = int(targets[k][l][1][m]/self.fract[0])
                    val_y = int(targets[k][l][0][m]/self.fract[0])
                    self.targets[k][val_x-1:val_x+2,val_y-1:val_y+2,j]=1
                    
        self.X = np.empty((self.batch_size,)+self.images[k].shape)
        self.Y = np.zeros((self.batch_size,)+output_shape+(len(targets[k]),))
        
    def __getitem__(self, index):
        
        while 1:
            tr = self.get_random_transform(self.images[self.keys[self.n]].shape)
            self.Y *=0
            
            for b in range(self.batch_size):
                
                self.X[b,:,:,:]=self.apply_transform(np.array(self.images[self.keys[self.n]]),tr)
                self.Y[b,:,:,:]=self.apply_transform(np.array(self.targets[self.keys[self.n]]),tr)
                self.n = (self.n+1) % self.n_samples
            yield self.X.copy(), self.Y.copy()


import matplotlib.pyplot as plt
import h5py

direct = '/home/sebastian/Documents/deeplabchop/projects/sebastian_evelien/data/first_half'
images = h5py.File('/media/sebastian/7B4861FD6D0F6AA2/inter_files','r')
#for f in os.listdir(direct):
#    images[f] = imageio.imread(os.path.join(direct,f))

batch_size = 4

with open('/home/sebastian/Documents/deeplabchop/projects/sebastian_evelien/labels.pkl','rb') as f:
    targets = pickle.load(f)
    
targets1 = {}
for k in targets.keys():
    targets1[k[81:]]=targets[k]

del targets

rand_keys = np.random.permutation(list(images.keys()))

margin = len(rand_keys)//10
train_keys=rand_keys[margin:]
test_keys = rand_keys[:margin]


train_gen = DeepLabGen(images,targets1,batch_size,keys = train_keys)
test_gen = DeepLabGen(images,targets1,batch_size, keys = test_keys)

net = DeepLab(3)
net.model.fit_generator(train_gen.__getitem__(0), epochs = 10,steps_per_epoch=500//batch_size,validation_data=test_gen.__getitem__(0), validation_steps = 50//batch_size)



for i in train_gen.__getitem__(1):
    print(i)
    break

plt.imshow(i[0][0,:,:,:].astype(np.int))
plt.scatter(*[(t*train_gen.fract[j]).astype(np.int) for j,t in enumerate(np.where(i[1][0,:,:,:].sum(2)))][::-1])
plt.show()

out1 = net.model.predict(i[0][np.newaxis,0,:,:,:])
plt.subplot(121)
plt.imshow(i[0][0,:,:,:].astype(np.int))
plt.subplot(122)
plt.imshow((out1[0,10:-10,10:-10,:].sum(2)))




