# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:32:49 2020

@author: maazk
"""

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath


rate = 2400000
dqpsk = np.genfromtxt("dqpsk.txt", dtype=complex128) 
d=int(len(dqpsk))


psd = np.abs(fft(np.abs(dqpsk),rate))


freq = 100000
angle = np.pi/2

data_x = np.linspace(0, d/rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x)+angle)

dqpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dqpsk_sync.append(dqpsk[i])  


unwrap = np.unwrap(np.angle(dqpsk_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(dqpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle)))

decode = []
for j in range(0,int(len(sync))):
    if (np.imag(sync[j]) > 60):
        decode.append("01")
    elif (np.imag(sync[j]) < -60 ):
        decode.append("10")
    elif (np.real(sync[j]) > 60):
        decode.append("00")   
    elif (np.real(sync[j]) < -60):
        decode.append("11")


final = np.transpose(decode)








#### Plots ####
# plt.plot(clock)
#plt.plot(np.abs(dqpsk))
# plt.plot(psd)
# plt.figure(1)
plt.plot(np.real(sync),np.imag(sync),"o")
# plt.figure(2)
# plt.polar(np.angle(sync),np.abs(sync),"o")

# plt.plot()
plt.grid()
plt.show()