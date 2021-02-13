import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath
sample_rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

plt.figure(1)
plt.plot((complex_data))
plt.grid()