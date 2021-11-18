#!/usr/bin/env python
# coding: utf-8

# In[8]:

import tensorflow as tf
import yaml
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from numpy import unique
from numpy import argmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from functions import *

config = None
with open('config.yml') as f:  
    config = yaml.load(f)


# data_path = config['paths']['path_small']  # Small_dataset
data_path = config['paths']['path_big']  # Big_dataset

# train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(**config['img_gen'])
validation_datagen = ImageDataGenerator(rescale=1./255)

# Date de training
train_ds = train_datagen.flow_from_directory(data_path + '/train',
                                             target_size=config['size'],
                                             shuffle=True,
                                             batch_size=config['model']['bs'],
                                             class_mode="binary")

# Date de validare
valid_ds = validation_datagen.flow_from_directory(data_path + '/valid',
                                                  target_size=config['size'],
                                                  shuffle=True,
                                                  batch_size=config['model']['bs'],
                                                  class_mode="binary")

print('train_ds: ', len(train_ds))
print('valid_ds: ', len(valid_ds))

# Date de testing
x_test, y_test = next(train_ds)
print(x_test.shape)
print(y_test)

labels = {0: 'COVID', 1: 'Normal'}
img_shape = (config['size'][0], config['size'][1], 3)


# Retea preantrenata
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(config['model']['n5'], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.add(BatchNormalization())
# model.summary()
conv_base.trainable = False

# API secvential
# model = Sequential()
# model.add(Conv2D(config['model']['n1'],
#                  config['model']['conv1'],
#                  activation='relu',
#                  input_shape=img_shape))
# model.add(MaxPool2D((2, 2)))
# model.add(Conv2D(config['model']['n2'], config['model']['conv2'], activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Conv2D(config['model']['n3'], config['model']['conv3'], activation='relu'))
# model.add(MaxPool2D((2, 2)))
# model.add(Conv2D(config['model']['n4'], config['model']['conv4'], activation='relu'))
# model.add(MaxPool2D((2, 2))10
# model.add(Flatten())
# # model.add(Dropout(0.5))
# model.add(Dense(config['model']['n5'], activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # model.add(BatchNormalization())
# # print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

history = model.fit(train_ds,
                    steps_per_epoch=len(train_ds),
                    validation_data=valid_ds,
                    validation_steps=len(valid_ds),
                    epochs=config['model']['n_epochs'])
model.save('covid.h5')

plot_acc_loss(history)

# Verificarea acuratetei reale
test_generator = validation_datagen.flow_from_directory(data_path + '/test',
                                                        target_size=config['size'],
                                                        batch_size=10,
                                                        class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator,
                                               steps=68)
print('test acc:', test_acc*100, '%')
