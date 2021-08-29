import math
import os
import time
# import Quandl
import pathlib

import pandas as pd
import pyarrow.parquet as pq

from machine_learning_with_python.utils.file_functions import get_dataframe_from_csv
# deprecated: cross validation is used for splitting up data sets
# svm = support vector machine. svm is able to perform regression
# from sklearn import preprocessing, cross_validation, svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))

_dir = pathlib.Path(HERE).resolve()

print(_dir.parent)

csv_file = f"{_dir.parent}/data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.csv"
parquet_file = f"{_dir.parent}/data/WIKI_PRICES_212b326a081eacca455e13140d7bb9db.parquet"

# Read CSV
# df = get_dataframe_from_csv(
#     csv_file
# )

# Pandas: Read Parquet
t1 = time.time()
df = pd.read_parquet(parquet_file, engine='pyarrow')
t2 = time.time()
delta_t = round((t2 - t1), 3)
print(f"Time it took = {delta_t} seconds\n")

# # PyArrow: Read Parquet
# # read in parquet file using pyarrow which has significant performance boost
# t1 = time.time()
# df = pq.read_table(parquet_file)
# t2 = time.time()
# delta_t = round((t2 - t1), 3)
# print(f"Time it took = {delta_t} seconds\n")

# We only need some of these categories for linear regression
df = df[
    [
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
    ]
]

# df["adj_open"] = pd.to_numeric(df["adj_open"], downcast="float")
# df["adj_open"] = pd.to_numeric(df["adj_open"], downcast="float")

# high minus low column
df["HL_PCT"] = (df["adj_high"] - df["adj_close"]) / df["adj_close"] * 100.0

# daily percent change
df["PCT_change"] = (df["adj_close"] - df["adj_open"]) / df["adj_open"] * 100.0

# We define a new datafram

df = df[["adj_close", "HL_PCT", "PCT_change", "adj_volume"]]


forecast_col = "adj_close"

# fill columns with NaN, but replace it with a real value. better than getting rid of data
df.fillna(-99999, inplace=True)

# round everything up to the nearest show number. We are trying to perdict 10% of the dataframe ( that's what the 0.1 is )
forecast_out = int(math.ceil(0.1 * len(df)))

# classifier ( the shift is forcasting the columns out negatively)
df["label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)  # Remove missing values.
# print(df.head())

# features = capital X
X = np.array(df.drop(['label'], 1))  # get everything except for label
# labels = lowercase y
y = np.array(df['label'])

# Now we are going to scale x
# in order to properly scale it, you need to scale them alongside all your other values (when training)
# SOURCE: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html
# SOURCE: https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
# Standardize a dataset along any axis.
# Center to the mean and component wise scale to unit variance.
# Read more in the User Guide.
X = preprocessing.scale(X)

# Redefine X (shift) - we don't need to do this because we are dropping the label already
# X = X[:-forecast_out+1]  # the point of where we were able to forecast the out plus +1
df.dropna(inplace=True)

y = np.array(df['label'])

# print(len(X), len(y))

# training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 20% of the data

# classifier definition and fit it
clf = LinearRegression(n_jobs=-1)  # choice A
# clf = svm.SVR() # change algorithm to   # choice B
clf.fit(X_train, y_train)  # train
accuracy = clf.score(X_train, y_train)  # test ( on seperate data, you want to use different data for this to make sure it actually works )

print(f"accuracy = {accuracy}\n")  # 0.000595491194672948 ( not very accurate )

# we are now ready to test, train, and predict

# /Users / malcolm / dev / universityofprofessorex / machine - learning - with-python
# ticker        date   open   high    low  close      volume  ex - dividend  split_ratio   adj_open   adj_high    adj_low  adj_close  adj_volume
# 0      A  1999 - 11 - 18  45.50  50.00  40.00  44.00  44739900.0          0.0          1.0  31.041951  34.112034  27.289627  30.018590  44739900.0
# 1      A  1999 - 11 - 19  42.94  43.00  39.81  40.38  10897100.0          0.0          1.0  29.295415  29.336350  27.160002  27.548879  10897100.0
# 2      A  1999 - 11 - 22  41.31  44.00  40.06  44.00   4705200.0          0.0          1.0  28.183363  30.018590  27.330562  30.018590   4705200.0
# 3      A  1999 - 11 - 23  42.50  43.63  40.25  40.25   4274400.0          0.0          1.0  28.995229  29.766161  27.460188  27.460188   4274400.0
# 4      A  1999 - 11 - 24  40.13  41.94  40.00  41.06   3464400.0          0.0          1.0  27.378319  28.613174  27.289627  28.012803   3464400.0
