# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:09:40 2020

@author: maazk
"""


import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath

sample_rate = 2400000
rate = 2400000
ook = np.genfromtxt("ook.txt", dtype=complex128) 
d=len(ook)
magnitude = np.abs(ook)


data_range = 2*d

######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(ook ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(ook ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(ook ))))





######## Removing freq offset #######
f_offset = 595000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*ook


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))





























freq = 104160+10
angle = 0*(np.pi)/4



data_x_2 = np.linspace(0, d/rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x_2+angle))



### clock Sync ###
ook_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        ook_sync.append(ook[i])   



### Decode ####
use = np.abs(ook_sync)
recovery = []
for j in range (0,1010):
        if ((use[j] > 47)):
            recovery.append("1")
        else:
            recovery.append("0")
        
decode = np.transpose(recovery[6:1010])  
 

 
# print(np.angle((ook[1011])))



############ Ploting ####################

### Clock vs ABS Data #####
plt.plot(clock,'b')
plt.plot(magnitude,'r')
plt.title("Clock vs |OOK|")
plt.ylabel("Amlitude")
plt.xlabel("Time ")




plt.figure(1)
plt.plot(data_x,psd,'r')
plt.plot(data_x,psd_offset,'b')
# plt.ylim(0,80)
# plt.xlim(-200000,150000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("Power Spectral Density")
plt.grid()


plt.figure(2)
plt.plot(fft_1,'b')
plt.xlim(-20000,sample_rate)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("FFT Of DBPSK")
plt.grid()



### Constalations #####
plt.figure(1)
plt.plot(np.real(ook),np.imag(ook),"o")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.title("Non Carrier Synced(OOK)")
plt.grid()

plt.figure(2)
plt.plot(np.real(ook_sync),np.imag(ook_sync),"o")
plt.title("Carrier Synced(OOK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()

plt.figure(3)
plt.polar(np.angle(ook_sync ),np.abs(ook_sync ),"o")
plt.title("Carrier Synced(OOK)")



plt.show()


#####################################################################################
# ^Error: Signal not found.$
# 01011110 01000101 01110010 01110010 01101111 01110010 00111010 00100000 01010011 01101001 01100111 01101110 01100001 01101100 00100000 01101110 01101111 01110100 00100000 01100110 01101111 01110101 01101110 01100100 00101110 00100100
########################################################################################