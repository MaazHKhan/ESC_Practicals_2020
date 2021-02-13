# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 15:20:41 2020

@author: alekm
"""


import numpy as np
from matplotlib import pyplot as plt
from pylab import *

sample_rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

huffman = complex_data[103000:133000]

# plt.figure(1)
# plt.plot(abs(complex_data))
# plt.grid()


d=int(len(huffman))

# plt.figure(1)
# plt.plot(abs(huffman))

magnitude = np.abs(huffman)

data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(huffman ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(huffman ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(huffman ))))


# ######## Removing freq offset #######
# f_offset = 0
# # f_offset = 342365
# clock_range = np.linspace(0,d/sample_rate,d)
# clock1 = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
# centered = clock1*huffman


# ######## Code for PSD centered  #########
# fft3 = np.fft.fft(centered,data_range)
# psd_shift2 = np.fft.fftshift(fft3)
# psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


# fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
# fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))







######################################
#### Resampling
#####################################

freq = 100000
angle = np.pi

data_x_2 = np.linspace(0, d/sample_rate, d)
clock =0.45*np.cos((2*np.pi*freq*data_x_2+angle-0.5))

huffman_sync = []
for i in range (d-27820,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        huffman_sync.append(huffman[i])  


unwrap = np.unwrap(np.angle(huffman_sync))



diff_angle = np.diff(unwrap)

sync = np.abs(huffman_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 0*1j*(np.sin(diff_angle)*1)) # -np.exp((1*np.pi/4.5)*1j)



decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0):
        decode.append("0")   
    elif (sync[j] < 0):
        decode.append("1")
        
samooosa = []
for i in range(0 ,len(decode),5): 
    samooosa.append(decode[i:i+5])
            
# decode = []
# for j in range(0,int(len(koelie))):
#     if (koelie[j] > 0):
#         decode.append("0")   
#     elif (koelie[j] < 0):
#         decode.append("1")

plt.figure(2)
# u = np.linspace(-0.7,0.7,20)
# zero = [0]*20
# mpl.plot(u,zero,'--')
# mpl.plot(zero,u,'--')
plt.grid()
# plt.plot(np.real(sync), np.imag(sync), 'o')
plt.plot(huffman)
plt.plot(clock)