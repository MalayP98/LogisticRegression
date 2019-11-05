import numpy as np
import pandas as pd
import math

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target


def train_test_split(input, target, ratio):
    shuffle_indices = np.random.permutation(input.shape[0])
    test_size = int(input.shape[0]*ratio)
    train_indices = shuffle_indices[:test_size]
    test_indices = shuffle_indices[test_size:]
    xTrain = input[train_indices]
    yTrain = target[train_indices]
    xTest = input[test_indices]
    yTest = target[test_indices]
    return xTrain, yTrain, xTest, yTest


class FeatureScaling:
    def __init__(self, data, type='Normalization'):
        self.data = data
        self.type = type

    def Scaling(self):

        if self.type == 'Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - min(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data

        elif self.type == 'Standardization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (np.var(self.data[:, i]))
            return self.data

        elif self.type == 'Mean_Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data


class LogisticRegression:
    def __init__(self, data, learning_rate):
        self.data = data
        self.learning_rate = learning_rate
        self.theta = np.random.rand(self.data.shape[1] + 1, 1)
        self.data = np.c_[np.ones((self.data.shape[0], 1)), self.data]
        print(self.theta, self.data)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))


    def linear_function(self):
        self.output = np.dot(self.data, self.theta)
        print(self.output)
        return self.output

    def train(self, target):
        for i in range(0,1000):
            self.error = self.sigmoid(self.linear_function()) - target
            """print("sigmoid o/p is \n",self.sigmoid(self.linear_function()))"""
            self.gradient = (np.dot(self.error.T, self.data)).T
            self.theta = self.theta - self.gradient*self.learning_rate
            """print("error \n", self.error)
            print("gradient \n", self.gradient)
            print("theta \n", self.theta)"""
        print("FINAL")
        print("error \n", self.error)
        print("gradient \n", self.gradient)
        print("theta \n", self.theta)

    def predict(self, input):
        print("Predicted values are...")
        return self.sigmoid(np.dot(input, self.theta))
        print(self.sigmoid(np.dot(input(), self.theta)))



"""xTest = np.array([[0,0],[0,1],[1,0],[1,1]])
target = np.array([[0],[0],[0],[1]])
obj = LogisticRegression(xTest,0.1)
obj.train(target)
obj.predict(xTest)"""

featureScaling = FeatureScaling(x, type = 'Normalization')
x = featureScaling.Scaling()

xTrain, yTrain, xTest, yTest = train_test_split(x,y,0.8)

logistic = LogisticRegression(xTrain, 0.1)
logistic.train(yTrain)

pred = logistic.predict(xTest)





