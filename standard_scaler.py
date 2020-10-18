import pandas as pd
from sklearn.preprocessing import minmax_scale, StandardScaler
import numpy as np
import timeit
from Bench import Bench

df = pd.read_csv("resources/skin.csv")

scaler = StandardScaler()

def standard_scaler(df):
    df[['R','G', 'B']] = scaler.fit_transform(df[['R','G', 'B']])

    return df

def cls(df):
    def wrap():
        return standard_scaler(df)

    return wrap

bench = Bench()

bench.times(cls(df), 30)

df = standard_scaler(df)

df.to_csv("standard_scaler-skin-py.csv", index=False)
