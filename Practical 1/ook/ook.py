# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:09:40 2020

@author: maazk
"""


import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import binascii
from scipy.fft import fft
# import cmath

rate = 2.4e6
ook = np.genfromtxt("ook.txt", dtype=complex128) 
d=len(ook)


### Power Spectral Density ####
psd = np.abs(fft(np.abs(ook),2400000))


freq = 104168
angle = 1*(np.pi)/4

# mag_ook = np.abs(ook)
# data_y = np.fft.fftshift(20*np.log10((abs(np.fft.fft(mag_ook)))))
# phase = np.fft.fftshift(20*np.log10(abs(np.fft.fft(np.angle(ook)))))

data_x = np.linspace(0, d/rate, d)
clock = 127*np.cos((2*np.pi*freq*data_x)+angle)



### clock Sync ### 
## re-sampling at the corect frequecy sample when
## cosine intesects the spliced data signal but only sample one 
ook_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        ook_sync.append(ook[i])   


### Decode #### threshold detection to 
use = np.abs(ook_sync)
recovery = []
for j in range (0,1010):
        if ((use[j] > 47)):
            recovery.append("1")
        else:
            recovery.append("0")
        
decode = np.transpose(recovery)  
 
# decode = []
# for k in range(0,1010)  :
#     decode.append(recovery[k])
#     if (k%8 == 0):
#         decode.append(" ")

# # value = int(recovery)






# values = bytearray(recovery)
# data_a2b = binascii.b2a_uu(decode)
# print(data_a2b)









#### Plots ####
# plt.plot(clock)
# plt.plot(np.abs(ook))
plt.plot((psd))

# plt.plot(decode)
# plt.plot(np.abs(ook_sync))

# plt.plot(np.real(ook),np.imag(ook),"o")
# plt.plot(np.real(ook_sync),np.imag(ook_sync),"o")

plt.grid()
plt.show()


