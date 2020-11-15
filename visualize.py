import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

sys.argv[1]

d1 = np.mean(pd.read_csv("resources/bench_output1.csv"), axis=0)
d2 = np.mean(pd.read_csv("resources/bench_output2.csv"), axis=0)
d3 = np.mean(pd.read_csv("resources/bench_output3.csv"), axis=0)
d4 = np.mean(pd.read_csv("resources/bench_output4.csv"), axis=0)
d5 = np.mean(pd.read_csv("resources/bench_output5.csv"), axis=0)
d6 = np.mean(pd.read_csv("resources/bench_output6.csv"), axis=0)

ds = pd.DataFrame(np.array([d1, d2, d3, d4, d5, d6]).T, columns=[1,2,3,4,5,6])

plt.figure(0)
plt.plot(ds.columns, ds.T[0], 'b')
plt.xlabel("Liczba procesorów")
plt.ylabel("Czas wykonania [ms]")
plt.savefig('resources/' + str(sys.argv[1]) + "1.png")

plt.figure(1)
plt.plot(ds.columns, abs(ds.T[0] / ds.T[0][1]), 'b')
plt.xlabel("Liczba procesorów")
plt.ylabel("Przyspieszenie")
plt.savefig('resources/' + str(sys.argv[1]) + "2.png")
