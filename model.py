#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Conv2D
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.readdata import train_generator, validation_generator, \
                     batchnum_train, batchnum_validation


def buildmodel():
	model = Sequential()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25), (0,0))))
	model.add(Conv2D(24, (5,5), strides=(2,2),activation="relu"))
	model.add(Conv2D(36, (5,5), strides=(2,2),activation="relu"))
	model.add(Conv2D(48, (5,5), strides=(2,2),activation="relu"))
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(Conv2D(64, (3,3), activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	
	model.compile(loss='mse', optimizer='adam')
	return model


def plotloss(history_object):
	"""plot the training and validation loss for each epoch.
	"""
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.savefig('training_curve.png')


def training():
	model = buildmodel()
	history_object = model.fit_generator(
	    train_generator, 
	    steps_per_epoch = batchnum_train,
	    epochs=5,
	    verbose=1,
	    validation_data = validation_generator,
	    validation_steps = batchnum_validation, 
	)
	model.save('model.h5')
	plotloss(history_object)


def main():
	training()


if __name__ = '__main__':
	main()
