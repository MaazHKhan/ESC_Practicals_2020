# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:32:15 2020

@author: maazk
"""


import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath


sample_rate = 2400000
dbpsk = np.genfromtxt("dbpsk.txt", dtype=complex128) 
d=int(len(dbpsk))




magnitude = np.abs(dbpsk)

data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(dbpsk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(dbpsk ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(dbpsk ))))





######## Removing freq offset #######
f_offset = 100000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*dbpsk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))















######################################
#### Resampling
#####################################

freq = 96147
angle = 2.22187

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x_2+angle))

dbpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dbpsk_sync.append(dbpsk[i])  


unwrap = np.unwrap(np.angle(dbpsk_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(dbpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 0*1j*(np.sin(diff_angle)*1))

decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 26):
        decode.append("0")   
    elif (sync[j] < 24):
        decode.append("1")


final = np.transpose(decode[6:1010])

points = []

for j in range(0,int(len(sync))):
    if (sync[j] > 60):
        points.append(-1)   
    elif (sync[j] < -60):
        points.append(1)



################################################################################
################################################################################


####### Plots FFTs and PSDs #########
plt.figure(1)
plt.plot(data_x,psd,'r')
plt.plot(data_x,psd_offset,'b')
# plt.ylim(0,80)
plt.xlim(-200000,150000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("Power Spectral Density")
plt.grid()


# plt.figure(2)
# plt.plot(fft_1,'b')
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT Of DBPSK")
# plt.grid()


# plt.figure(3)
# plt.plot(fft_2)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.title("FFT with frequency offset")
# plt.grid()


# plt.figure(4)
# plt.plot(np.angle(dbpsk ))
# plt.ylabel("Phase (rad)")
# plt.xlabel("samples")
# plt.grid()

# plt.figure(5)
# plt.plot(fft_4)
# plt.title("Centered FFT")
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.grid()

# plt.figure(6)
# plt.plot(fft_5)
# plt.title("Centered FFT")
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.grid()




# ### Clocks
# plt.plot(clock,'b')
# plt.plot(np.abs(dbpsk),'r')
# plt.title("Clock vs |DBPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()









###### Constatlations
# plt.figure(1)
# plt.polar(np.angle(sync[0:len(points)]),np.abs(points),"o")
# plt.title("Carrier Synced(DBPSK)")

# plt.figure(2)
# plt.plot(np.real(sync/75),np.imag(sync/75),"o")
# plt.title("Carrier Synced(DBPSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()

# plt.figure(3)
# plt.plot(np.real(dbpsk),np.imag(dbpsk),"o")
# plt.title("Non Carrier Synced(DBPSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()















##############################################################################
# 00100000 01101111 01110100 01101000 01100101 01110010 00100000 01101000 01100001 01101110 01100100 00101100 00100000 01111001 01101111 01110101 00100000 01101000 01100001 01110110 01100101 00100000 01100100 01101001 01100110 01100110 01100101 01110010 01100101 01101110 01110100 00100000 01100110 01101001 01101110 01100111 01100101 01110010 01110011 00101110 00100100

# ^On the other hand, you have different fingers.$
###############################################################################