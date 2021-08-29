# SOURCE: https://www.kaggle.com/dkmostafa/predicting-stock-market-using-linear-regression
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import pathlib
import time

from typing import Any, List, Sequence, Union

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from rich import print as rich_print
from rich.console import Console
from rich.table import Table
from sklearn import linear_model, preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

console = Console()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

HERE = os.path.abspath(os.path.dirname(__file__))

_dir = pathlib.Path(HERE).resolve()

rich_print(_dir.parent)

input_dir = f"{_dir.parent}/data/kaggle/nyse"

console.log(os.listdir(input_dir))

# Union[Union[int, str], float] == Union[int, str, float]
def prepare_data(
    df: pd.DataFrame, forecast_col: str, forecast_out: int, test_size: float
) -> List[Union[Union[Sequence, Any], List]]:
    label = df[forecast_col].shift(
        -forecast_out
    )  # creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]])  # creating the feature array
    X = preprocessing.scale(X)  # processing the feature array
    X_lately = X[
        -forecast_out:
    ]  # creating the column i want to use later in the predicting method
    X = X[:-forecast_out]  # X that will contain the training and testing
    label.dropna(inplace=True)  # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=test_size
    )  # cross validation

    response = [X_train, X_test, Y_train, Y_test, X_lately]
    return response


# Pandas: Read Parquet
t1 = time.time()
df = pd.read_csv(f"{input_dir}/prices.csv")  # loading the csv file
df = df[df.symbol == "GOOG"]  # choosing stock symbol
t2 = time.time()
delta_t = round((t2 - t1), 3)
rich_print(f"Time it took = {delta_t} seconds\n")


forecast_col = "close"  # choosing which column to forecast
forecast_out = 5  # how far to forecast
test_size = 0.2  # the size of my test set

X_train, X_test, Y_train, Y_test, X_lately = prepare_data(
    df, forecast_col, forecast_out, test_size
)  # calling the method were the cross validation and data preperation is in

learner = linear_model.LinearRegression(
    n_jobs=-1
)  # initializing linear regression model

learner.fit(X_train, Y_train)  # training the linear regression model
score = learner.score(X_test, Y_test)  # testing the linear regression model

forecast = learner.predict(X_lately)  # set that will contain the forecasted data

response = {}  # creating json object
response["test_score"] = score
response["forecast_set"] = forecast

rich_print("")
# console.log(response, log_locals=True)
console.log(response)
rich_print("")


# Any results you write to the current directory are saved as output.
