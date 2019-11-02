import numpy as np
import pandas as pd
import maths

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def findMin(data):
    return min(data)

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target

class FeatureScaling:
    def __init__(self, data, type='Standardization'):
        if type=='Normalization':
            return (data - data.min())/(data.max()-data.min())
        elif type == 'Standardization':
            return (data-data.mean())




class LogisticRegression:




















