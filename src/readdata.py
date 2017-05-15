#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def readimage(imname):
    im = cv2.imread(imname)
    if im is None:
        print("{} can not read".format(imname))
    return im


def readdata(csv_file, path):
    car_imagenames = []
    steering_angles = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        lines = [line for line in reader]
        for row in lines[1:]:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.1 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # add image names and angles to data set
            car_imagenames.extend([
                os.path.join(path, os.path.basename(row[0]).strip()),
                os.path.join(path, os.path.basename(row[1]).strip()),
                os.path.join(path, os.path.basename(row[2]).strip()),
                   ])
            steering_angles.extend([
                steering_center, 
                steering_left, 
                steering_right,
                   ])
    return car_imagenames, steering_angles


def generator(samples, labels, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples, labels = shuffle(samples, labels)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = labels[offset:offset+batch_size]
            for batch_sample in batch_samples:
                images.append(readimage(batch_sample))
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train, y_train = shuffle(X_train, y_train)
            yield X_train, y_train


allsamples = []
alllabels = []
datarootfolder = './data'
datafolderlist = os.listdir(datarootfolder)
print(datafolderlist)
for datafolder in datafolderlist:
    if os.path.isfile(os.path.join(datarootfolder, datafolder, 'driving_log.csv')):
        samples, labels = readdata(
            os.path.join(datarootfolder, datafolder, 'driving_log.csv'), 
            os.path.join(datarootfolder, datafolder, 'IMG'))
        allsamples.extend(samples)
        alllabels.extend(labels)

X_train, X_test, y_train, y_test = train_test_split(allsamples, alllabels, test_size=0.33, random_state=42)


samplenum_train = len(y_train)
samplenum_validation = len(y_test)
batchnum_train = int(samplenum_train/128)
batchnum_validation = int(samplenum_validation/128)

train_generator = generator(X_train, y_train, batch_size=128)
validation_generator = generator(X_test, y_test, batch_size=128)

