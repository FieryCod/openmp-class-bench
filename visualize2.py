import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

P_NAME = sys.argv[1]

d1 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "12260.csv"), axis=0)
d2 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "13260.csv"), axis=0)
d3 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "14260.csv"), axis=0)
d4 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "15260.csv"), axis=0)
d5 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "16260.csv"), axis=0)
d6 = np.mean(pd.read_csv("resources/bench_outputB_" + P_NAME + "17260.csv"), axis=0)

ds = pd.DataFrame(np.array([d1, d2, d3, d4, d5, d6]).T, columns=[1,2,3,4,5,6])


plt.figure(0)
plt.plot([12260, 13260, 14260, 15260, 16260, 17260], ds.T[0], 'b')
plt.xlabel("Blocki")
plt.ylabel("Czas wykonania [ms]")
plt.savefig('resources/' + str(sys.argv[1]) + "B1.png")

plt.figure(1)
plt.plot([12260, 13260, 14260, 15260, 16260, 17260], abs(ds.T[0] / ds.T[0][1]) * 100, 'b')
plt.xlabel("Blocki")
plt.ylabel("Przyspieszenie")
plt.savefig('resources/' + str(sys.argv[1]) + "B2.png")
