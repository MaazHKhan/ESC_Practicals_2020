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

magnitude = np.abs(dqpsk)


sample_rate = 2400000
data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(dqpsk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(dqpsk ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(dqpsk ))))





######## Removing freq offset #######
f_offset = -150000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*dqpsk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))






































freq = 100000
angle = 1.7

data_x_2 = np.linspace(0, d/rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x)+angle)

dqpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dqpsk_sync.append(dqpsk[i])  


unwrap = np.unwrap(np.angle(dqpsk_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(dqpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) * np.exp((-11*np.pi/10)*1j)


points =[]
for j in range(0,int(len(sync))):
    if (sync[j] > 25):
        points.append("00")   
    elif (sync[j] < -25):
        points.append("11")     
    elif (np.imag(sync[j]) > 1j* 25):
        points.append("01")
    elif (np.imag(sync[j]) < 1j* 25):
        points.append("10")

final = np.transpose(points[1:1010])





############ Ploting ####################

####### Plots FFTs and PSDs #########
# plt.figure(1)
# plt.plot(data_x,psd,'r')
# plt.plot(data_x,psd_offset,'b')
# # plt.ylim(0,80)
# # plt.xlim(-200000,150000)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("Power Spectral Density")
# plt.grid()


# plt.figure(2)
# plt.plot(fft_1,'b')
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT Of DQPSK")
# plt.grid()














# ### Clocks
# plt.plot(clock,'b')
# plt.plot(np.abs(dqpsk),'r')
# plt.title("Clock vs |DQPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()

# plt.figure(1)

# plt.plot(np.real(dqpsk)/80,np.imag(dqpsk)/75,"o")
# plt.title(" Non Carrier Synced(DQSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()



# plt.figure(2)
# plt.polar(np.angle(sync[0:1010]),np.abs(sync[0:1010]),"o")
# plt.title("Carrier Synced(DQSK)")
# plt.plot()

plt.show()









#######################################
# ^What if there were no hypothetical questions?
# 01011110 01010111 01101000 01100001 01110100 00100000 01101001 01100110 00100000 01110100 01101000 01100101 01110010 01100101 00100000 01110111 01100101 01110010 01100101 00100000 01101110 01101111 00100000 01101000 01111001 01110000 01101111 01110100 01101000 01100101 01110100 01101001 01100011 01100001 01101100 00100000 01110001 01110101 01100101 01110011 01110100 01101001 01101111 01101110 01110011 00111111