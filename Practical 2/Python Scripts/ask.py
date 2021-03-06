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

file = open("data1.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

plt.figure(1)
plt.plot(abs(complex_data))

ask = complex_data[98000:125000]
magnitude = np.abs(ask)
d=int(len(ask))

plt.figure(18)
plt.plot(abs(ask))

data_range = 2*d


file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()

I_data_raw = interleaved_data[0:len(interleaved_data):2] 
Q_data_raw = interleaved_data[1:len(interleaved_data):2] 
fs = 2.4e6



I_samples = ((I_data_raw-127.5)/127.5)
Q_samples = (Q_data_raw-127.5)/127.5
binsb = 98000
binsf = 125000

y_r = I_samples[binsb:binsf]
raw2 = Q_samples[binsb:binsf]


complex_data = (I_samples + 1j*Q_samples)
complex_data1=complex_data[binsb:binsf]
nbins = len(complex_data1)
t = np.arange(0, nbins/fs, 1/fs)



######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(ask,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 10*np.log10(np.abs(np.fft.fft(np.abs(ask),sample_rate)))


######## Code for fft bins  #########
fft_2  = 10*np.log10(np.abs(np.fft.fft(np.abs(ask))))



######## Removing freq offset #######
# f_offset = 346000
f_offset = -251496*1.846#-464300
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = (np.cos(2*np.pi*-251496*t*1.846153846)+1j*np.sin(2*np.pi*-251496*t*1.846153846))*ask


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))



########### Resampling ###########

freq2 = 55645*1.846153846#102734 #55645*1.846153846
phase = -0.9*0
clock2 = 1*np.cos(2*np.pi*freq2*clock_range+phase)


sync = []
for i in range (0,d-1):
    if (clock2[i] > clock2[i-1] and clock2[i] > clock2[i+1]):
        sync.append(centered[i])



unwrap = np.unwrap(np.angle(sync))
# plt.plot(unwrap)
diff_angle = np.diff(unwrap) - (1*np.pi)*0
# plt.plot(diff_angle)
normalise = np.abs(sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + (1j*(np.sin(diff_angle)))) *np.exp((np.pi*1)*1j)


# normalise = np.transpose(sync) *np.exp((0)*1j)
 


recovery = []
for k in range (0,int(len(normalise))):
        if (normalise[k] < -0.2):
            recovery.append("00")
        elif (normalise[k] > -0.2 and normalise[k] < -0.1):
            recovery.append("01")
        elif (normalise[k] >  -0.1 and normalise[k] < 0.2):
            recovery.append("11")
        elif (normalise[k] > 0.2):
            recovery.append("10")



# const = []
# for k in range (0,int(len(normalise))):
#         if (normalise[k] > -1.2 and np.real(normalise[k]) < -0.5  ):
#             const.append(-1)
#         elif (normalise[k] > -0.5 and normalise[k] < 0)  :
#             const.append(-0.5)
#         elif (normalise[k] >  0 and normalise[k] < 0.5 ):
#             const.append(0.5)
#         elif (normalise[k] > 0.5 and normalise[k] < 1.2 ):
#             const.append(1)


decode = np.transpose(recovery) 



########## Down sampling Plots ##########

# plt.plot(clock2,'b')
# plt.plot(np.abs(centered),'r')
# plt.title("Clock vs |ASK|")
# plt.xlabel("Time")
# plt.xlabel("Amplitude")
# plt.plot(np.unwrap(np.angle(centered)))
# plt.grid()


######## Plots FFTs and PSDs #########
plt.figure(1)
plt.plot(data_x,psd,'r')
plt.plot(data_x,psd_offset,'b')
# plt.xlim(-323700,91000)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("Power Spectral Density")
plt.grid()


plt.figure(2)
plt.plot(fft_1)
plt.xlim(-20000,sample_rate)
# plt.ylim(-2,80)
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency (Hz)")
plt.title("FFT of ASK")
plt.grid()


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

# plt.figure(3)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.grid()


plt.figure(4)
# plt.polar(0*np.angle(normalise),np.abs(normalise),"o")
# plt.ylim(-5,5)
plt.plot(np.real(normalise),np.imag(normalise),"o")
plt.title("Carrier Synced(ASK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


# plt.figure(3)
# plt.plot(np.real(sync_2),np.imag(sync_2),"o")
# plt.grid()

# plt.figure(5)
# plt.polar(np.angle(const),np.abs(const),"o")
# plt.title("Synced normalised (ASK)")