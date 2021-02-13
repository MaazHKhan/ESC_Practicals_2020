# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:23:24 2020

@author: maazk
"""

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath



dbpsk = np.genfromtxt("ask.txt", dtype=complex128) 
d=int(len(dbpsk))



rate = 2400000
magnitude = np.abs(dbpsk)

psd= np.abs(20*np.log10(((np.fft.fft(dbpsk)))))
fft = 20*np.log10(np.abs(np.fft.fft(np.abs(dbpsk),rate)))
fft_bins = 20*np.log10((np.abs(np.fft.fft(magnitude))))


freq = 94924
angle = 2.22187

data_x = np.linspace(0, d/rate, d)
clock = 1*np.cos((2*np.pi*freq*data_x)+angle)

dbpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dbpsk_sync.append(dbpsk[i])  


unwrap = np.unwrap(np.angle(dbpsk_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(dbpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) 

normalise = sync*-np.exp((1*np.pi/6)*1j)

recovery = []
for k in range (0,int(len(normalise))):
        if (normalise[k] > -100 and np.real(normalise[k]) < -30  ):
            recovery.append("00")
        elif (normalise[k] > -30 and normalise[k] < 0)  :
            recovery.append("01")
        elif (normalise[k] >  0 and normalise[k] < 40 ):
            recovery.append("11")
        elif (normalise[k] > 40 and normalise[k] < 100 ):
            recovery.append("10")



decode = np.transpose(recovery) 



#### Plots ####

#### Clocks
# plt.plot(clock)
# plt.plot(np.abs(dbpsk))


#### FFT and PSD
# plt.plot(fft)
# plt.grid()

# figure(1)
# x_axis = np.linspace(0,2400,rate)
# plt.figure(1)
# plt.plot(x_axis,fft)
# # plt.plot(fft)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (kHz)")
# plt.title("FFT of input data (4-ASK)")
# # plt.xlim(0,rate,1000000)
# plt.grid()


# figure(2)
# plt.plot(psd)



# figure(3)
# plt.plot(fft_bins)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Sample Bins")
# plt.title("FFT of input data (4-ASK)")
# plt.grid()


# figure(4)
# plt.plot(np.angle((dbpsk)))
# plt.ylabel("Phase (radians)")
# plt.xlabel("Sample Bins")
# plt.title("Phase vs samples (4-ASK)")
# plt.grid()





###### Constatlations
# plt.figure(1)
# plt.polar(np.angle(sync[0:len(points)]),np.abs(points),"o")
# plt.title("Carrier Synced(DBPSK)")

plt.figure(2)
plt.plot(np.real(normalise),np.imag(normalise),"o")
plt.title("Carrier Synced(DBPSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()

# plt.figure(3)
# plt.plot(np.real(dbpsk),np.imag(dbpsk),"o")
# plt.title("Non Carrier Synced(DBPSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()

plt.show()