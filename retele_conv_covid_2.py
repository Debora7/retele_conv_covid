#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
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
import yaml
import matplotlib.pyplot as plt

config = None
with open('config.yml') as f:  
    config = yaml.load(f)
    
data_path = config['path_small'] # Small_dataset
# data_path = config['path_big'] # Big_dataset
img_size = config['size']
bs = config['bs']

train_datagen = ImageDataGenerator( rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

# train_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Date de training
train_ds = train_datagen.flow_from_directory(data_path + '/train',
                                             target_size=img_size,
                                             shuffle=True,
                                             batch_size=bs,
                                             class_mode="binary")

# Date de validare
valid_ds = validation_datagen.flow_from_directory(data_path + '/valid',
                                                  target_size=img_size,
                                                  shuffle=True,
                                                  batch_size=bs,
                                                  class_mode="binary")

# Date de testing
x_test, y_test = next(train_ds)
print(x_test.shape)
print(y_test)

labels = {0: 'COVID', 1: 'Normal'}
img_shape = (img_size[0], img_size[1], 3)

from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)
conv_base.summary()

# API secvential
model = Sequential()
model.add(Conv2D(config['n1'], config['conv1'], activation='relu', input_shape=img_shape))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(config['n2'], config['conv2'], activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(config['n3'], config['conv3'], activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(config['n4'], config['conv4'], activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(config['n5'], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.add(BatchNormalization())
print(model.summary())

# Retea preantrenata
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
conv_base.trainable = False

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

nr_ep = config['n_epochs']
size_dataset_valid = len(valid_ds)
size_dataset_train = len(train_ds)

history = model.fit(train_ds, steps_per_epoch=size_dataset_train, validation_data=valid_ds, validation_steps=size_dataset_train, epochs=nr_ep)
model.save('covid.h5')

# Functia de plotare a acuratetei si a functiei loss
def plot_acc_loss(result):
    acc = result.history['accuracy']
    loss = result.history['loss']
    val_acc = result.history['val_accuracy']
    val_loss = result.history['val_loss']
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()


plot_acc_loss(history)

# Verificarea acuratetei reale
test_generator = validation_datagen.flow_from_directory(data_path + '/test', target_size=(64, 64), batch_size=10, 
                                                        class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=size_dataset_valid)
print('test acc:', test_acc*100, '%')

