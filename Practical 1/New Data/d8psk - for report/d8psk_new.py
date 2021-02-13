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

# plt.plot(abs(complex_data))
# 
ask = complex_data[181620:181620+18000]
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
f_offset = -405000

p=0
clock_range = np.linspace(0,d/sample_rate,d)
clock = np.cos(2*np.pi*f_offset*clock_range+p) + 1j*np.sin(2*np.pi*f_offset*clock_range+p)
centered = clock*ask


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))



########### Resampling ###########

freq2 = 94930
phase = 0
clock2 = np.cos(2*np.pi*freq2*clock_range)+phase


sync = []
for i in range (0,d-1):
    if (clock2[i] > clock2[i-1] and clock2[i] > clock2[i+1]):
        sync.append(centered[i])

normalise = np.exp((0)*1j) * np.transpose(sync) 
 



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






final = np.transpose(decode) 

    
# plt.figure(1)
# plt.plot(np.unwrap(np.angle(centered)))
# plt.grid()


# plt.plot(clock2)
# plt.plot(np.abs(centered))



# plt.plot(np.unwrap(np.angle(centered)))
# plt.grid()




# plt.plot(clock2,'b')
# plt.plot(np.abs(centered),'r')
# plt.title("Clock vs |D8PSK|")
# plt.xlabel("Time")
# plt.xlabel("Amplitude")

######## Plots FFTs and PSDs #########
# plt.figure(1)
# plt.plot(data_x,psd,'r')
# plt.plot(data_x,psd_offset,'b')
# # plt.xlim(-sample_rate/2,sample_rate/2)
# plt.ylim(0,100)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("Power Spectral Density")
# plt.grid()


# plt.figure(2)
# plt.plot(fft_1)
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT of D8PSK")
# plt.grid()


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


# plt.figure(1)
# plt.plot(np.real(ask),np.imag(ask),"o")
# plt.ylabel("Imaginary")
# plt.xlabel("Real")
# plt.grid()



# plt.figure(2)
# plt.plot(np.real(centered),np.imag(centered),"o")
# plt.grid()

plt.figure(3)
plt.plot(np.real(centered),np.imag(centered),"o")
plt.title("Non Carrier Synced(D8PSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()






# plt.figure(4)
# plt.plot(np.real(normalise),np.imag(normalise),"o")
# plt.grid()
# plt.polar(0*np.angle(normalise),np.abs(normalise),"o")

# plt.figure(5)
# plt.polar(np.angle(sync),np.abs(sync),"o")
# plt.title(("Synced non normalised)")