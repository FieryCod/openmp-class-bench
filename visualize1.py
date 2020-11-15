import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

P_NAME = sys.argv[1]

d1 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "20.csv"), axis=0)
d2 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "40.csv"), axis=0)
d3 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "60.csv"), axis=0)
d4 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "80.csv"), axis=0)
d5 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "100.csv"), axis=0)
d6 = np.mean(pd.read_csv("resources/bench_outputT_" + P_NAME + "120.csv"), axis=0)

ds = pd.DataFrame(np.array([d1, d2, d3, d4, d5, d6]).T, columns=[1,2,3,4,5,6])

plt.figure(0)
plt.plot([20, 40, 60, 80, 100, 120], ds.T[0], 'b')
plt.xlabel("Thready")
plt.ylabel("Czas wykonania [ms]")
plt.savefig('resources/' + str(sys.argv[1]) + "T1.png")

plt.figure(1)
plt.plot([20, 40, 60, 80, 100, 120], abs(ds.T[0] / ds.T[0][1]) * 100, 'b')
plt.xlabel("Thready")
plt.ylabel("Przyspieszenie")
plt.savefig('resources/' + str(sys.argv[1]) + "T2.png")
