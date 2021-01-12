#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

omp = [23193, 22319, 21435, 20967, 20870, 20661, 21106, 20652, 22060, 21508, 20656, 21120, 20877, 20722, 21011, 20641, 20976, 20681, 20303, 20646, 20472, 20648, 20289, 20574, 20919, 20991, 20575, 20310, 20648, 20386]
hybrid = [734, 713, 667, 637, 655, 682, 656, 624, 631, 632, 655, 675, 674, 684, 671, 670, 660, 766, 754, 659, 660, 662, 663, 666, 671, 668, 692, 668, 692, 649]
mpi = [935, 990, 1899, 1662, 918, 1029, 1090, 1085, 1086, 1087, 1086, 1097, 1097, 1002, 999, 1004, 1126, 1332, 1450, 1116, 1114, 1143, 1101, 1102, 1119, 1205, 1320, 1509, 1101, 1199]

plt.figure(0)
plt.xlabel("Numer uruchomienia")
plt.ylabel("Czas wykonania log(ms)")
plt.scatter(range(0, 30), np.log(omp), label='OpenOMP')
plt.scatter(range(0, 30), np.log(hybrid), label='Hybrid')
plt.scatter(range(0, 30), np.log(mpi), label='OpenMPI')
plt.legend()
plt.savefig('resources/all.png')
plt.show()

plt.figure(0)
plt.xlabel("Numer uruchomienia")
plt.ylabel("Czas wykonania (ms)")
plt.scatter(range(0, 30), omp, label='OpenOMP')
plt.scatter(range(0, 30), hybrid, label='Hybrid')
plt.scatter(range(0, 30), mpi, label='OpenMPI')
plt.legend()
plt.savefig('resources/all1.png')
plt.show()
