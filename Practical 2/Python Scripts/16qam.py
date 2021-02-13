# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:22:09 2020

@author: maazk
"""

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath

sample_rate = 2400000
rate = 2400000
#######################################################################
########################### Reading in data ########################### 
#######################################################################
file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples
#######################################################################
#######################################################################

qam = complex_data[304052:329936]
d=int(len(qam))
magnitude = np.abs(qam)

# plt.plot(complex_data)



data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(qam ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(qam ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(qam ))))


######## Removing freq offset #######
f_offset = 848772
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*qam


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))





freq = 93743-7
angle = -0.08*np.pi/2
data_x_2 = np.linspace(0,d/sample_rate,d)
clock = 1*np.cos(2*np.pi*freq*data_x_2+angle)

qam_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        qam_sync.append(centered[i])  






unwrap = np.unwrap(np.angle(qam_sync))
diff_angle = -1*np.diff(unwrap) + np.pi/2
sync = np.abs(qam_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) *np.exp((0.1*np.pi/4)*1j) 

# plt.plot(-1*np.unwrap(np.angle(qam)))
# points =[]
# for j in range(0,int(len(sync))):
#     if (sync[j] > 0.25):
#         points.append("00")   
#     elif (sync[j] < -0.25):
#         points.append("11")     
    # elif (np.imag(sync[j]) > 1j* 0.25):
#         points.append("01")
#     elif (np.imag(sync[j]) < 1j* 0.25):
#         points.append("10")

# final = np.transpose(points[3:384])

plt.figure(1)
plt.plot(np.unwrap(np.angle(qam_sync)))
plt.plot(np.unwrap(np.angle(sync)))

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
# plt.title("FFT Of qam")
# plt.grid()

# # ### Clocks
# plt.figure(3)
# plt.plot(clock,'b')
# plt.plot(np.abs(qam),'r')
# plt.title("Clock vs |qam|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()

# plt.figure(4)
# plt.plot(np.real(qam),np.imag(qam),"o")
# plt.title(" Non Carrier Synced(16-QAM)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


plt.figure(5)
plt.plot(np.real(qam_sync),np.imag(qam_sync),"o")
plt.title("Carrier Synced(16-QAM)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


# plt.figure(6)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.title("Carrier Synced(16-QAM) Demodulated")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


plt.figure(6)
plt.polar(np.angle(qam_sync),np.abs(qam_sync),"o")
plt.title("Carrier Synced(DQSK)")
plt.plot()

plt.show()


