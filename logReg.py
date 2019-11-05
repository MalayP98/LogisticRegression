import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target
y = y.reshape(y.shape[0], 1)


def confusion_matrix(input, target):
    true_positive = 0
    true_negative = 0
    flase_positive = 0
    flase_negative = 0
    for i in range(input.shape[0]):
        if input[i] >= 0.5:
            input[i] = 1
        else:
            input[i] = 0

    for i in range(target.shape[0]):
        if input[i] == target[i]:
            if input[i] == 1:
                true_positive += 1
            else:
                true_negative += 1
        elif input[i] != target[i]:
            if input[i] == 0:
                flase_negative += 1
            else:
                flase_positive += 1
    print("Confusion Matrix: ", np.array([[true_positive,flase_positive],[flase_negative, true_negative]]))
    print("True Positive = ", true_positive)
    print("False Positive = ", flase_positive)
    print("True Negative = ", true_negative)
    print("False Negative = ", flase_negative)

def train_test_split(input, target, ratio):
    shuffle_indices = np.random.permutation(input.shape[0])
    test_size = int(input.shape[0] * ratio)
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def linear_function(self):
        self.output = np.dot(self.data, self.theta)
        return self.output

    def train(self, target, iteration):
        for i in range(0, iteration):
            self.error = self.sigmoid(self.linear_function()) - target
            self.gradient = (np.dot(self.error.T, self.data)).T
            self.theta = self.theta - (self.gradient * self.learning_rate)
        print("FINAL")
        print("error \n", self.error)
        print("gradient \n", self.gradient)
        print("theta \n", self.theta)

    def predict(self, input):
        print("Predicted values are...")
        input = np.c_[np.ones((input.shape[0], 1)), input]
        print(input.shape, self.theta.shape)
        return self.sigmoid(np.dot(input, self.theta))
        print(self.sigmoid(np.dot(input(), self.theta)))


featureScaling = FeatureScaling(x, type='Normalization')
x = featureScaling.Scaling()

xTrain, yTrain, xTest, yTest = train_test_split(x, y, 0.8)

logistic = LogisticRegression(xTrain, 0.1)
logistic.train(yTrain, 1000)

pred = logistic.predict(xTest)
confusion_matrix(pred, yTest)