# -- coding: utf-8 --
"""
Created on Sat Sep 12 16:33:28 2020

@author: Ryan Rabe
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

bpsk = complex_data[127710:150996]

# plt.figure(1)
#plt.plot(abs(complex_data))
# plt.grid()


d=int(len(bpsk))

# plt.figure(1)
# plt.plot(abs(bpsk))

magnitude = np.abs(bpsk)

data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(bpsk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(bpsk ),sample_rate)))

# plt.plot(fft_1)

######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(bpsk ))))


######## Removing freq offset #######
f_offset = -276679
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*bpsk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))




# plt.plot(np.unwrap(np.angle(centered)))


######################################
#### Resampling
#####################################

freq = 104159
angle = np.pi/2.5

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = np.cos((2*np.pi*freq*data_x_2+angle))

bpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        bpsk_sync.append(centered[i])  



rotated = np.transpose(bpsk_sync)*-np.exp((1*np.pi/4)*1j)

unwrap = np.unwrap(np.angle(bpsk_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(bpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle)*0)) 

# pre_decode = np.zeros(int(len(sync))+1)
# ang_sync = np.angle(sync)
# for x in range(0,int(len(sync))):
#     pre_decode[x+1] = pre_decode[x] + ang_sync[x]
    

    
# sync_2 = np.abs(bpsk_sync[0:int(len(pre_decode))]) * (np.cos(pre_decode) + 1*1j*(np.sin(pre_decode)*0))     




decode = []
for j in range(0,int(len(rotated))):
    if (rotated[j] > 0):
        decode.append("0")   
    elif (rotated[j] < 0):
        decode.append("1")


final = np.transpose(decode)
points = []

for j in range(0,int(len(sync))):
    if (sync[j] > 0.2):
        points.append(-1)   
    elif (sync[j] < -0.2):
        points.append(1)



####### Plots FFTs and PSDs #########
# plt.figure(1)
# plt.plot(data_x,psd,'r')
# plt.plot(data_x,psd_offset,'b')
# # plt.ylim(0,80)
# # plt.xlim(-200000,150000)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("Power Spectral Density")
# plt.grid()


# plt.figure(3)
# plt.plot(fft_2)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.title("FFT with frequency offset")
# plt.grid()


# plt.figure(4)
# plt.plot(np.angle(dbpsk ))
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


# ### Clocks
# plt.plot(clock,'b')
# plt.plot(np.abs(dbpsk),'r')
# plt.title("Clock vs |DBPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()


##### Constatlations
# plt.figure(7)
# plt.polar(np.angle(sync[0:len(points)]),np.abs(points),"o")
# plt.title("Carrier Synced(BPSK)")

plt.figure(8)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.title("Carrier Synced(BPSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()




plt.figure(9)
plt.plot(np.real(rotated),np.imag(rotated),"o")
plt.title("Carrier Synced(BPSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()



# ### Clocks
# plt.figure(3)
# plt.plot(clock,'b')
# plt.plot(np.abs(dbpsk),'r')
# plt.title("Clock vs |DBPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()






##############################################################################
# ^As long as there are tests, there will be prayer in schools.$
# 01011110 01000001 01110011 00100000 01101100 01101111 01101110 01100111 
# 00100000 01100001 01110011 00100000 01110100 01101000 01100101 01110010 
# 01100101 00100000 01100001 01110010 01100101 00100000 01110100 01100101 
# 01110011 01110100 01110011 00101100 00100000 01110100 01101000 01100101 
# 01110010 01100101 00100000 01110111 01101001 01101100 01101100 00100000 
# 01100010 01100101 00100000 01110000 01110010 01100001 01111001 01100101 
# 01110010 00100000 01101001 01101110 00100000 01110011 01100011 01101000 
# 01101111 01101111 01101100 01110011 00101110 00100100 00001010
##############################################################################