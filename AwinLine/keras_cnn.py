#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from os.path import isfile, isdir, join
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

import numpy as np
import xlwt
import base64
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

import lineprocess

people = []
nbatch_size = 128
nepochs = 48
image_size = (96, 96)
imageWidth = 96
imageHeight = 96


def LoadData(folderName):
    images = []
    labels = []
    num_classes = 0

    for dirname in os.listdir(fileDir + "/" + folderName):
        fullpath = join(fileDir + "/" + folderName, dirname)
        if isfile(fullpath):
            continue
        for file in os.listdir(fullpath):
            img = image.load_img(os.path.join(
                fullpath, file), target_size=image_size)
            img_array = image.img_to_array(img)
            # print(img_array)
            images.append(img_array)
            labels.append(num_classes)

        num_classes += 1

    data = np.array(images)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, num_classes)
    return data, labels, num_classes


def BulidModuel(num_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(
        imageWidth, imageHeight, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(5, 5),
                     activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.summary()  # 輸出網路結構
    print("compile.......")
    sgd = Adam(lr=0.0003)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def main():
    xTrain, xLabels, num_classes = LoadData("images")
    cnnModel = BulidModuel(num_classes)
    xTrain /= 255
    x_train, x_test, y_train, y_test = train_test_split(
        xTrain, xLabels, test_size=0.2)
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    tbCallbacks = callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    cnnModel.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs,
                 verbose=1, validation_data=(x_test, y_test), callbacks=[tbCallbacks])

    print("evaluate......")
    scroe, accuracy = cnnModel.evaluate(x_test, y_test, batch_size=nbatch_size)
    print('scroe:', scroe, 'accuracy:', accuracy)
    yaml_string = cnnModel.to_yaml()
    with open(os.path.join(fileDir, "") + 'keras_cnn.yaml', 'w') as outfile:
        outfile.write(yaml_string)

    cnnModel.save_weights(os.path.join(fileDir, "") + 'keras_cnn.h5')


def LoadPeople(folderName):
    global people
    for dirname in os.listdir(fileDir + "/" + folderName):
        people.append(dirname)


def predict(folderName, excelName):
    global people
    LoadPeople("images")

    with open(os.path.join(fileDir, "") + 'keras_cnn.yaml') as yamlfile:
        loaded_model_yaml = yamlfile.read()
    model = model_from_yaml(loaded_model_yaml)
    model.load_weights(os.path.join(fileDir, "") + 'keras_cnn.h5')

    sgd = Adam(lr=0.0003)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    pattern = xlwt.Pattern()
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN
    pattern.pattern_fore_colour = 2
    style = xlwt.XFStyle()
    style.pattern = pattern
    workbook = xlwt.Workbook()
    worksheet = workbook.add_sheet('My Sheet')
    row = 0
    for dirname in os.listdir(fileDir + "/" + folderName):
        fullpath = join(fileDir + "/" + folderName, dirname)
        if isfile(fullpath):
            continue

        for file in os.listdir(fullpath):
            img = image.load_img(os.path.join(
                fullpath, file), target_size=image_size)
            img_array = image.img_to_array(img)

            x = np.expand_dims(img_array, axis=0)
            x = preprocess_input(x)
            result = model.predict_classes(x, verbose=0)

            worksheet.write(row, 0, dirname)
            worksheet.write(row, 1, people[result[0]])
            row = row + 1
    workbook.save(os.path.join(fileDir, "") + excelName + '.xls')


def NormalizationImage(folderName):
    for dirname in os.listdir(fileDir + "/" + folderName):
        fullpath = join(fileDir + "/" + folderName, dirname)
        if isfile(fullpath):
            continue

        for file in os.listdir(fullpath):
            fp = open(os.path.join(fullpath, file), "rb")
            img = fp.read()
            fp.close()
            try:
                rgb = lineprocess.NormalizationImage(
                    "data:image/jpeg;base64," + base64.b64encode(img))
                img = Image.fromarray(rgb, 'RGB')
            except Exception as e:
                print(e)
                continue

            os.remove(os.path.join(fullpath, file))
            img.save(os.path.join(fullpath, file))


if __name__ == '__main__':
    NormalizationImage('images')
    # NormalizationImage('known')
    # NormalizationImage('unknown')
    # main()
    #predict('known', 'Known_Keras')
    #predict('unknown', 'Unknown_Keras')
