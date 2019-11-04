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

    def sigmoid(self):
        pass

    def linear_function(self, instance):
        self.output = np.dot(self.data[instance], self.theta)
        return self.output




