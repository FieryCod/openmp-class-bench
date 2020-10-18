import time
import numpy as np

class Bench:

    def __init__(self):
        self.ops = []

    def times(self, func, ops_times = 30):
        for x in range(30):
            tic = time.perf_counter()

            func()

            tac = time.perf_counter()

            self.ops.append(tac - tic)


        print('Mean of elapsed time for ' + str(ops_times) + ' execs took ' + str(np.mean(self.ops) * 1000) + ' ms')
