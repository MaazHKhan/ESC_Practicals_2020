# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:23:24 2020

@author: maazk
"""
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath
sample_rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

# plt.figure(1)
# plt.plot(abs(complex_data))
# plt.grid()


ask = complex_data[90742:90742+11840]
magnitude = np.abs(ask)
d=int(len(ask))

data_range = 2*d



######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(ask,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(ask),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(ask))))



######## Removing freq offset #######
f_offset = 350000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*ask


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))



########### Resampling ###########


freq2 = 94922
phase = 5.21863*0
clock2 = 1*(np.cos(2*np.pi*freq2*clock_range)+phase)


sync = []
for i in range (0,d-1):
    if (clock2[i] > clock2[i-1] and clock2[i] > clock2[i+1]):
        sync.append(centered[i])

normalise = np.transpose(sync) * np.exp((3.9)*1j)*1/0.6
 


recovery = []
for k in range (0,int(len(normalise))):
        if (normalise[k] > -1.2 and np.real(normalise[k]) < -0.5  ):
            recovery.append("00")
        elif (normalise[k] > -0.5 and normalise[k] < 0)  :
            recovery.append("01")
        elif (normalise[k] >  0 and normalise[k] < 0.5 ):
            recovery.append("11")
        elif (normalise[k] > 0.5 and normalise[k] < 1.2 ):
            recovery.append("10")


decode = np.transpose(recovery[0:469]) 

    




# plt.plot(clock2)
# plt.plot(np.abs(centered))



# plt.plot(np.unwrap(np.angle(centered)))
# plt.grid()






######## Plots FFTs and PSDs #########
# plt.figure(1)
# plt.plot(data_x,psd,'r')
# plt.plot(data_x,psd_offset,'b')
# plt.ylim(0,80)
# # plt.xlim(-323700,91000)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("Power Spectral Density")
# plt.grid()
# plt.show()


# plt.figure(2)
# plt.plot(fft_1)
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT with frequency offset")
# plt.grid()
# plt.show()


# plt.figure(3)
# plt.plot(fft_2)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.title("FFT with frequency offset")
# plt.grid()


# plt.figure(4)
# plt.plot(np.angle(ask))
# plt.ylabel("Phase (rad)")
# plt.xlabel("samples")
# plt.grid()

# plt.figure(5)
# plt.plot(fft_4)
# plt.title("Centered FFT")
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.grid()

# plt.figure(6)
# plt.plot(fft_5)
# plt.title("Centered FFT")
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.grid()





######## Plots constalations #########


plt.figure(1)
plt.plot(np.real(ask),np.imag(ask),"o")
plt.title("Non Carrier Synced(ASK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()



# plt.figure(2)
# plt.plot(np.real(centered),np.imag(centered),"o")
# plt.grid()
# plt.show()

# plt.figure(3)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.grid()
# plt.show()


# plt.figure(4)
# # plt.polar(0*np.angle(normalise),np.abs(normalise),"o")
# plt.plot(np.real(normalise),np.imag(normalise),"o")
# plt.grid()
# plt.show()


# plt.figure(5)
# plt.polar(np.angle(sync),np.abs(sync),"o")
# plt.title(("Synced non normalised)")

################################################################################################
# ^My mother used to make me walk the plank when I was younger because we couldn't afford a dog.$
# 01011110 01001101 01111001 00100000 01101101 01101111 01110100 01101000 01100101 01110010 00100000 01110101 01110011 01100101 01100100 00100000 01110100 01101111 00100000 01101101 01100001 01101011 01100101 00100000 01101101 01100101 00100000 01110111 01100001 01101100 01101011 00100000 01110100 01101000 01100101 00100000 01110000 01101100 01100001 01101110 01101011 00100000 01110111 01101000 01100101 01101110 00100000 01001001 00100000 01110111 01100001 01110011 00100000 01111001 01101111 01110101 01101110 01100111 01100101 01110010 00100000 01100010 01100101 01100011 01100001 01110101 01110011 01100101 00100000 01110111 01100101 00100000 01100011 01101111 01110101 01101100 01100100 01101110 00100111 01110100 00100000 01100001 01100110 01100110 01101111 01110010 01100100 00100000 01100001 00100000 01100100 01101111 01100111 00101110 00100100