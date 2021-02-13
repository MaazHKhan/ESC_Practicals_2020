# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:51:16` 2019

@author: Maaz
"""
# import csv
import numpy as np
import matplotlib.pyplot as plt
# from pylab import *
# from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter


sample_rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

# plt.figure(1)
# plt.plot(abs(complex_data))
# plt.grid()



data = complex_data[14397:39130]
 

N = 1024  # output size

rrcos = rrcosfilter(N*4, 0.8, 1, 24)[1]



new_data = np.convolve(rrcos, data) # Waveform with PSF

mag = np.abs(new_data)
data_x = -1*(20*np.log10((abs(np.fft.fft(mag)))))





# plt.plot(np.angle(new_data))
# plt.plot(rrcos)
# plt.plot(new_data,'r')
plt.plot(mag,'b')
plt.show()