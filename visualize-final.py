import matplotlib.pyplot as plt
import numpy as np

mpi = [3481, 1834,1167, 1218]
omp = [64267, 31437,20941, 21731]
hyb = [1747, 834,673, 705]

x = [2, 4, 6, 8]

plt.ylabel("Czas wykonania log(ms)")
plt.xlabel("Ilość procesów")
plt.plot(x, np.log(mpi), label='OpenMPI')
plt.plot(x, np.log(omp), label='OpenOMP')
plt.plot(x, np.log(hyb), label='Hybrid')
plt.legend()
plt.savefig('resources/cwlog.png')
plt.show()

plt.ylabel("Czas wykonania (ms)")
plt.xlabel("Ilość procesów")
plt.plot(x, (mpi), label='OpenMPI')
plt.plot(x, (omp), label='OpenOMP')
plt.plot(x, (hyb), label='Hybrid')
plt.legend()
plt.savefig('resources/cw.png')
plt.show()

plt.ylabel("Czas wykonania (ms)")
plt.plot('OpenOMP', (omp[2]), 'o', label='OpenOMP')
plt.plot('OpenMPI', (mpi[2]),'o', label='OpenMPI')
plt.plot('Hybrid', (hyb[2]), 'o', label='Hybrid')
plt.legend()
plt.savefig('resources/cwBest.png')
plt.show()

mpip = [3481, 1634,1167, 1218]
ompp = [62267, 30437,20941, 21731]
hybp = [1747, 834,673, 705]

for i in range(0, 4):
    mpip[i] = mpi[0] - mpi[i]
    ompp[i] = omp[0] - omp[i]
    hybp[i] = hyb[0] - hyb[i]

plt.ylabel("Przyspieszenie log(ms)")
plt.xlabel("Ilość procesów")
plt.plot(x, np.insert(np.log(mpip[1:]), 0, 0, axis=0), label='OpenMPI')
plt.plot(x, np.insert(np.log(ompp[1:]), 0, 0, axis=0), label='OpenOMP')
plt.plot(x, np.insert(np.log(hybp[1:]), 0, 0, axis=0), label='Hybrid')
plt.legend()
plt.savefig('resources/plog.png')
plt.show()

plt.ylabel("Przyspieszenie (ms)")
plt.xlabel("Ilość procesów")
plt.plot(x, (mpip), label='OpenMPI')
plt.plot(x, (ompp), label='OpenOMP')
plt.plot(x, (hybp), label='Hybrid')
plt.legend()
plt.savefig('resources/p.png')
plt.show()

plt.ylabel("Przyspieszenie (ms)")
plt.plot('OpenOMP', (omp[2]-omp[2]), 'o', label='OpenOMP')
plt.plot('OpenMPI', (omp[2]-mpi[2]),'o', label='OpenMPI')
plt.plot('Hybrid', (omp[2]-hyb[2]), 'o', label='Hybrid')
plt.legend()
plt.savefig('resources/pBest.png')
plt.show()