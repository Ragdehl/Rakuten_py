# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 23:07:27 2021

@author: Edgar
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('images\image_train',
                                                 target_size = (64, 64), 
                                                 batch_size = 32, 
                                                 class_mode = 'categorical') 


