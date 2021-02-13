# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:17:00 2020

@author: maazk
"""

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
# from commpy.modulation import QAMModem
from commpy.filters import rrcosfilter


sample_rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

m_filter = complex_data[14397:39130]

mag = np.abs(m_filter)

d=int(len(m_filter))


magnitude = np.abs(m_filter)
plt.figure(1)
plt.plot((m_filter))
plt.grid()

data_x = (20*np.log10((abs(np.fft.fft(mag)))))


#Scipy Website Commpy Library root raised Cosines

rrcos = rrcosfilter(1*2, 0.5, 1, 2400000)[1] 
f_sample_factor=11
sinc_func=np.sin(np.arange(0,f_sample_factor,1)*np.pi/f_sample_factor)



rollof=10
rrcos_2=3*np.sinc(np.arange(-rollof,rollof,0.5)*np.pi/rollof)
filterout_t=np.correlate((m_filter),sinc_func)
mag = np.abs(filterout_t)
angle = np.angle(filterout_t)
unwrap = np.unwrap(angle)

fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(filterout_t ),sample_rate)))


######################################
#### Resampling
#####################################

freq = 100000
angle = 2*0

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = 2.5*np.cos((2*np.pi*freq*data_x_2+angle))

new_data_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        new_data_sync.append(filterout_t[i])  


unwrap = np.unwrap(np.angle(new_data_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(new_data_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1*1j*(np.sin(diff_angle)*1)) #* -np.exp((1*np.pi/4.5)*1j)



decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0):
        decode.append("0")   
    elif (sync[j] < 0):
        decode.append("1")


final = np.transpose(decode[3:235])


# plt.figure(7)
# plt.plot(clock)
# plt.plot(abs(filterout_t))


# plt.figure(2)
# plt.plot(rrcos, label='Root Raised Cosine')

# plt.figure(3)
# plt.plot(data_x)
# plt.grid()

# plt.figure(5)
# plt.plot(filterout_t)
# plt.plot((rrcos_2))


# plt.figure(6)
# plt.plot(fft_1)



# plt.figure(11)
# plt.plot(rrcos_2)
# plt.ylabel('Amplitude')
# plt.xlabel('Frequency')
# plt.title('Matched Filter')
# plt.grid()
# plt.legend()



# plt.figure(8)
# plt.plot(np.real(new_data_sync),np.imag(new_data_sync),"o")
# # plt.ylim(-1,1)
# plt.title("Carrier Synced(huffman)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


# plt.figure(9)
# plt.plot(np.real(sync),np.imag(sync),"o")
# # plt.ylim(-1,1)
# plt.title("Carrier Synced(huffman)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()

