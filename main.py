#!/usr/bin/env python
import sys

def progress(current, total):
    percent = 100*(current+1)/total
    sys.stdout.write('\rLoading: %d%s [%s> %s]' % (percent, '%', '='*percent, (99-percent)*' ' ))
    sys.stdout.flush()

import matplotlib.pyplot as plt
import matplotlib.image as mat_img
import numpy as np
import cv2

class Image:
    '''Constructor'''
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_name = image_path.split('/')[-1]
        self.data = self.set_data()
        self.label = self.image_name.split('.')[0]
    '''set image content to data'''
    def set_data(self):
        data = mat_img.imread(self.image_path)
        data = cv2.resize(data, dsize=(200, 200))
        return np.array(data.flatten())
    '''Return image reshaped'''
    def get_image(self):
        return self.data.reshape(200, 200, 3)
    '''Draw image'''
    def draw_image(self):
        print('label: %s'%(self.label))
        plt.imshow(self.get_image())
        plt.show()

from os import listdir
class Images:
    '''Constructor'''
    def __init__(self, data_path=None):
        self.data = []
        if (data_path != None):
            self.load_images(data_path)
    '''Add an image to data list'''
    def add(self, image, total=1):
        progress(len(self.data), total)
        self.data.append(image)
    '''Load multiple images from data path'''
    def load_images(self, data_path):
        for image in listdir(data_path):
            self.add(Image(data_path+'/'+image), len(listdir(data_path)))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class Model_Wrapper:
    '''contructor'''
    def __init__(self, high, size, images=None):
        self.data = []
        self.labels = []
        self.high = high
        self.size = size
        if (images != None):
            self.load_data(images)
    '''data loader'''
    def load_data(self, images):
        selector = np.random.randint(self.high, size=self.size)
        self.data = [ images.data[i].data for i in selector ]
        self.labels = [ images.data[i].label for i in selector ]
    '''train data'''
    def train(self, train_size):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.data, self.labels, train_size=train_size)
    '''build a random forest model'''
    def build_random_forest_model(self):
        self.train(0.8)
        self.model_name = 'Random forest'
        self.model = RandomForestClassifier(n_jobs=2, random_state=0)
        self.model.fit(self.xtrain, self.ytrain)
    '''build a k-nearest neighbors model'''
    def build_knn_model(self):
        self.train(0.8)
        self.model_name = 'k-nearest neighbors'
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.xtrain, self.ytrain)
    '''build a logistic regression model'''
    def build_logistic_regression_model(self):
        self.train(0.8)
        self.model_name = 'logistic regression'
        self.model = LogisticRegression()
        self.model.fit(self.xtrain, self.ytrain)
    def accuracy(self):
        self.accuracy = self.model.score(self.xtest, self.ytest)
        print('Your model accuracy is: %.2f%s.\n' %(self.accuracy*100, '%'))
    '''make a prediction'''
    def predict(self, image):
        return self.model.predict([image])

import configargparse

p = configargparse.ArgParser()
p.add('-d', '--data-dir', required=True, help='Path to data folder')
p.add('-m', '--model', required=True, help=' randomForest | knn | logisticRegression')


if __name__ == '__main__':

    options = p.parse_known_args()

    data_path = options[0].data_dir
    model_name = options[0].model

    images = Images(data_path)

    model = Model_Wrapper(len(listdir(data_path))-1, (len(listdir(data_path))-1)/2, images)
    
    if model_name == 'randomForest':
        print('Building random Forest model.')
        model.build_random_forest_model()
    elif model_name == 'knn':
        print('Building KNN model.')
        model.build_knn_model()
    elif model_name == 'logisticRegression':
        print('Building logistic regression model.')
        model.build_logistic_regression_model()

    model.accuracy()
