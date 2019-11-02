import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target




















