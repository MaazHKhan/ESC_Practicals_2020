# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:11:22 2020

@author: maazk
"""

import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath

rate = 2.4e6
ask = np.genfromtxt("ask.txt", dtype=complex128) 
d=len(ask)


psd = np.abs(fft(np.abs(ask),2400000))
# plt.plot(np.abs(psd))
# psd = np.abs(np.fft.fft(np.abs(ask),2400000))
# plt.plot((psd))


freq = 99993
angle = 0.5

# data_y = np.fft.fftshift(20*np.log10((abs(np.fft.fft(mag_ask)))))
# phase = np.fft.fftshift(20*np.log10(abs(np.fft.fft(np.angle(ask)))))

data_x = np.linspace(0, 24887/rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x)+angle)

ask_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        ask_sync.append(ask[i])   


normalisede_angle = np.unwrap(np.angle(ask_sync))

diff_angle = np.diff(normalisede_angle)+(np.pi/4)+np.pi

new_ask_final = np.abs(ask_sync[0:1035]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle)*0))


nor_diff_ang = np.arange(0,3.14,np.pi/1036)
the_angle = np.angle(new_ask_final)

for j in range(0,1035):
    nor_diff_ang[j+1]= the_angle[j] + nor_diff_ang[j]
    
    
final = 160*np.abs(ask_sync[0:1036]) * (np.cos(nor_diff_ang) + 1j*(np.sin(nor_diff_ang)*0))    




# ### Decode ####

recovery = []
for k in range (0,1036):
        if (final[k] > -100 and final[k] < -50  ):
            recovery.append("00")
        elif (final[k] > -50 and final[k] < 0  ):
            recovery.append("01")
        elif (final[k] >  0 and final[k] < 50  ):
            recovery.append("11")
        elif (final[k] > 50 and final[k] < 100  ):
            recovery.append("10")
                       
            
decode = np.transpose(recovery)   
        
   
        

#### Plots ####
# plt.plot(clock)
# plt.plot(np.abs(ask))

# plt.plot(np.real( ask_sync),np.imag(ask_sync),"o")
plt.plot(np.real(final),np.imag(final),"o")

plt.grid()
plt.show()

