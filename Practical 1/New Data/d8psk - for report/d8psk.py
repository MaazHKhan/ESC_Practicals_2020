# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:23:24 2020

@author: maazk
"""
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath



d8psk = np.genfromtxt("d8psk.txt", dtype=complex128) 
rate = 2400000
d=int(len(d8psk))
magnitude = np.abs(d8psk)


psd= np.abs(20*np.log10(((np.fft.fft(d8psk)))))
fft = 20*np.log10(np.abs(np.fft.fft(np.abs(d8psk),rate)))
fft_bins = 20*np.log10((np.abs(np.fft.fft(magnitude))))

freq = 94930+3
angle = 0*0.9712621092193812

data_x = np.linspace(0, d/rate, d)
clock = 1*np.cos((2*np.pi*freq*data_x)+angle)

d8psk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        d8psk_sync.append(d8psk[i])  


unwrap = np.unwrap(np.angle(d8psk_sync))

diff_angle = np.diff(unwrap) + (1*np.pi/7)

sync = np.abs(d8psk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + (1j*(np.sin(diff_angle)))) * np.exp((0*np.pi/4)*1j)

sync_angle = []
sync_mag = []


for j in range(0,int(len(sync))):
    if (np.angle(sync[j]) > -3.12 and np.angle(sync[j]) < -2):
        sync_angle.append(0*np.pi) 
        
    elif (np.angle(sync[j]) > 0.161*np.pi and np.angle(sync[j]) < 0.343*np.pi):
        sync_angle.append(0.25*np.pi) 
    elif (np.angle(sync[j]) > 0.4*np.pi and np.angle(sync[j]) < 0.6*np.pi):
        sync_angle.append(0.5*np.pi) 
    elif (np.angle(sync[j]) > 0.626*np.pi and np.angle(sync[j]) < 0.866*np.pi):
        sync_angle.append(0.75*np.pi)
  
    elif (np.angle(sync[j]) > 0.866*np.pi and np.angle(sync[j]) < 1.12*np.pi):
        sync_angle.append(1*np.pi)
        
    elif (np.angle(sync[j]) > -0.343*np.pi and np.angle(sync[j]) < -0.161*np.pi):
        sync_angle.append(1.25*np.pi)
    elif (np.angle(sync[j]) > -0.6*np.pi and np.angle(sync[j]) < -0.4*np.pi):
        sync_angle.append(1.5*np.pi)
    elif (np.angle(sync[j]) >-1.150  and np.angle(sync[j]) < -0.349 ):
        sync_angle.append(1.75*np.pi)


for k in range(0,int(len(sync_angle))):
    sync_mag.append(1)


decode = []
for j in range(0,int(len(sync))):
    if (np.angle(sync[j]) > -3.12 and np.angle(sync[j]) < -2):
        decode.append("000") 
        
    elif (np.angle(sync[j]) > 0.161*np.pi and np.angle(sync[j]) < 0.343*np.pi):
        decode.append("001") 
    elif (np.angle(sync[j]) > 0.4*np.pi and np.angle(sync[j]) < 0.6*np.pi):
        decode.append("011") 
    elif (np.angle(sync[j]) > 0.626*np.pi and np.angle(sync[j]) < 0.866*np.pi):
        decode.append("010")
  
    elif (np.angle(sync[j]) > 0.866*np.pi and np.angle(sync[j]) < 1.12*np.pi):
        decode.append("110")
        
    elif (np.angle(sync[j]) > -0.343*np.pi and np.angle(sync[j]) < -0.161*np.pi):
        decode.append("111")
    elif (np.angle(sync[j]) > -0.6*np.pi and np.angle(sync[j]) < -0.4*np.pi):
        decode.append("101")
    elif (np.angle(sync[j]) >-1.150  and np.angle(sync[j]) < -0.349 ):
        decode.append("100")






phase = np.angle(sync)
magnitude = np.abs(sync)
soprted = np.sort(phase)
final= np.transpose(decode)
print(np.angle((d8psk[1038])))


tran = np.transpose(final)



#### Clock vs ABS Data #####
# plt.plot(clock)
# plt.plot(abs(d8psk))
# plt.ylabel("Amlitude")
# plt.xlabel("Time (ms)")




#### FFT and PSD #####
# plt.plot(fft)
# plt.grid()

x_axis = np.linspace(0,2400,rate)
plt.figure(1)
plt.plot(x_axis,fft)
# 
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (kHz)")
plt.title("FFT of input data (d8psk)")
# plt.xlim(0,rate,1000000)
plt.grid()

# plt.figure(2)
# plt.plot(np.linspace(0,2400,25682),psd)
# plt.grid()

# plt.figure(3)
# plt.plot(fft_bins)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Sample Bins")
# plt.title("FFT of input data (d8psk)")
# plt.grid()

# plt.figure(4)
# plt.plot(np.angle((d8psk)))
# plt.ylabel("Phase (radians)")
# plt.xlabel("Sample Bins")
# plt.title("Phase vs samples (d8psk)")
# plt.grid()






#### Constalations #####
# plt.figure(1)
# plt.plot(np.real(d8psk),np.imag(d8psk))
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.title("Non Carrier Synced(d8psk)")
# plt.grid()

# plt.figure(2)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.title("Carrier Synced(D8PSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()

# plt.figure(3)
# plt.polar(sync_angle,np.abs(sync_mag),"o")
# plt.title("Carrier Synced(D8PSK)")

