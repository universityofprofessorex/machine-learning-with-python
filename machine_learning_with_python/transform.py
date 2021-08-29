import math
import os
# import Quandl
import pathlib

import pandas as pd

from machine_learning_with_python.utils.file_functions import get_dataframe_from_csv
# deprecated: cross validation is used for splitting up data sets
# svm = support vector machine. svm is able to perform regression
# from sklearn import preprocessing, cross_validation, svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import numpy as np
from machine_learning_with_python.utils.csv_to_parquet import to_parquet

HERE = os.path.abspath(os.path.dirname(__file__))

_dir = pathlib.Path(HERE).resolve()

print(_dir.parent)

df = get_dataframe_from_csv(
    f"{_dir.parent}/data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"
)

csv_file = f"{_dir.parent}/data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"
parquet_file = f"{_dir.parent}/data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.parquet"
sep = ","

to_parquet(csv_file, parquet_file, sep=",")
