# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 14:22:09 2020

@author: maazk
"""

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from scipy import signal

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

qam = complex_data[274516:298780]
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
f_offset = 660000+1000*26
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


freq = 99993-7
angle = 4*np.pi/2
data_x_2 = np.linspace(0,d/sample_rate,d)
clock = 1*np.cos(2*np.pi*freq*data_x_2+angle)

qam_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        qam_sync.append(centered[i])  


# ########################################
#  ################### PLL ###############
# ########################################
reduced_size = qam_sync
pllIn=np.angle(reduced_size)
predict=0
neww=np.arange(0,len(qam_sync),1)

for i in range(0,len(reduced_size)-11):
    predict=(np.average(np.diff(pllIn[i:i+12])))+pllIn[i+11]
    neww[i+11]=pllIn[i+11]-predict
    if (neww[i+11]>=np.pi/4 and  neww[i+11]<3*np.pi/4):
        pllIn[i+11]=neww[i+11]-np.pi/2
    elif (neww[i+11]>=-np.pi/4 and  neww[i+11]<np.pi/4):
        pllIn[i+11]=neww[i+11]+0
    elif (neww[i+11]>=-3*np.pi/4 and  neww[i+11]<-np.pi/4):
        pllIn[i+11]=neww[i+11]+np.pi/2
    else:
        pllIn[i+11]=neww[i+11]+np.pi
        
sync_x=np.cos(pllIn)+1j*np.sin(pllIn)*np.exp((1*np.pi/4)*1j)


sort = []
for i in range(0,len(sync_x)):
    if(np.real(sync_x[i]) > 0.25):
        sort.append(1)
    elif(np.real(sync_x[i]) < -0.5):
        sort.append(-1)
    elif(np.imag(sync_x[i]) > 0.94*1j):
        sort.append(1*1j)
    elif(np.imag(sync_x[i]) < -0.95*1j):
        sort.append(-1*1j)
        
sync = np.transpose(sort)*np.exp((1*np.pi/4)*1j)

# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.plot(np.unwrap(np.angle(sync)))
# plt.plot(np.unwrap(np.angle(qam_sync)))

unwrap = np.unwrap(np.angle(qam_sync))
diff_angle = np.diff(unwrap)
# sync = np.abs(qam_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) * np.exp((5.9*np.pi/4)*1j) 
# 
# plt.plot(np.unwrap(np.angle(new_angle)))
# plt.plot(np.unwrap(np.angle(diff_angle)))

points =[]
for j in range(0,int(len(sync))):
    if (np.real(sync[j]) > 0.5 and np.imag(sync[j]) < -0.4 ):
        points.append("10")   
    elif (np.real(sync[j]) < -0.5 and np.imag(sync[j]) < 0.2 ):
        points.append("00")     
    elif (np.real(sync[j]) > 0.5 and np.imag(sync[j]) >= 0 ):
        points.append("11")
    elif (np.real(sync[j]) < -0.5 and np.imag(sync[j]) > 0.2 ):
        points.append("01")

final = np.transpose(points)



############ Ploting ####################

####### Plots FFTs and PSDs #########
plt.figure(1)
plt.cla()
plt.plot(data_x,psd,'r')
plt.plot(data_x,psd_offset,'b')
# plt.ylim(0,80)
# plt.xlim(-200000,150000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("Power Spectral Density")
plt.grid()

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
# plt.title(" Non Carrier Synced(4-QAM)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


plt.figure(5)
plt.cla()
plt.plot(np.real(sync_x),np.imag(sync_x),'o')
plt.title("Carrier Synced(4-QAM) Demodulated")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


plt.figure(6)
plt.cla()
plt.plot(np.real(qam_sync),np.imag(qam_sync),"o")
plt.title("Offset Removed symblob Synced(4-QAM)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


# plt.figure(7)
# plt.polar(np.angle(sync),np.abs(sync),"o")
# plt.title("Carrier Synced(DQSK)")
# plt.plot()

plt.show()


