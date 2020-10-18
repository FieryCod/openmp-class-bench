import pandas as pd
from sklearn.preprocessing import minmax_scale
import numpy as np
import timeit
from Bench import Bench

df = pd.read_csv("resources/skin.csv")

def min_max_scale(df):
    df[['R','G', 'B']] = minmax_scale(df[['R','G', 'B']])

    return df

def cls(df):
    def wrap():
        return min_max_scale(df)

    return wrap

bench = Bench()

bench.times(cls(df), 30)

df = min_max_scale(df)

df.to_csv("min_max-skin-py.csv", index=False)
