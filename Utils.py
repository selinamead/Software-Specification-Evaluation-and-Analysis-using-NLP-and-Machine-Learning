from sklearn.base import BaseEstimator, TransformerMixin
from pandas.io.parsers import _get_col_names
import numpy as np
import pandas as pd

# class to allow the selection of specific columns of a dataframe.
# It converts a dataframe selection into numpy array to be able to apply later scikitlearn functions on them
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self,col_names):
        self.col_names = col_names
    def fit(self,X, y=None):
        return self
    def transform(self,X):
        return X[self.col_names].values


 