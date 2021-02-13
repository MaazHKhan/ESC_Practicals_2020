# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:23:24 2020

@author: maazk
"""
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath


rate = 2400000
d8psk = np.genfromtxt("d8psk.txt", dtype=complex128) 
d=int(len(d8psk))


psd = 20*np.log10(np.abs(fft(np.abs(d8psk),rate)))


freq = 94937
angle = 0

data_x = np.linspace(0, d/rate, d)
clock = 180*np.cos((2*np.pi*freq*data_x)+angle)

d8psk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        d8psk_sync.append(d8psk[i])  


unwrap = np.unwrap(np.angle(d8psk_sync))

diff_angle = np.diff(unwrap)+(5*np.pi/8)

sync = np.abs(d8psk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle)))

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
    sync_mag.append(100)


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







#### Plots ####
# plt.plot(clock)
# plt.plot(np.abs(d8psk))
# plt.plot(psd)

plt.figure(1)
plt.polar(phase, magnitude,"o")
plt.figure(2)
plt.polar(sync_angle, sync_mag,"o")
plt.figure(3)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.plot()

plt.grid()
plt.show()
