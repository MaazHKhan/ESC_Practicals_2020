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

dbpsk = complex_data[156126:180530]

# plt.figure(1)
# plt.plot(abs(complex_data))
# plt.grid()


d=int(len(dbpsk))

# plt.figure(1)
# plt.plot(abs(dbpsk))

magnitude = np.abs(dbpsk)

data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(dbpsk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(dbpsk ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(dbpsk ))))


######## Removing freq offset #######
f_offset = -87000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*dbpsk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))







######################################
#### Resampling
#####################################

freq = 99992
angle = 2.22187*0

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = np.cos((2*np.pi*freq*data_x_2+angle))

dbpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dbpsk_sync.append(dbpsk[i])  


unwrap = np.unwrap(np.angle(dbpsk_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(dbpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1*1j*(np.sin(diff_angle)*1)) * -np.exp((1*np.pi/4.5)*1j)



decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0.2):
        decode.append("1")   
    elif (sync[j] < -0.2):
        decode.append("0")


final = np.transpose(decode[1:680])

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
plt.figure(7)
plt.polar(np.angle(sync[0:len(points)]),np.abs(points),"o")
plt.title("Carrier Synced(DBPSK)")

plt.figure(8)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.ylim(-1,1)
plt.title("Carrier Synced(DBPSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()

# plt.figure(9)
# plt.plot(np.real(dbpsk),np.imag(dbpsk),"o")
# plt.title("Non Carrier Synced(DBPSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()







# ### Clocks
# plt.figure(3)
# plt.plot(clock,'b')
# plt.plot(np.abs(dbpsk),'r')
# plt.title("Clock vs |DBPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()



##############################################################################################
# ^Smoking will kill you, and bacon will kill you.  But, smoking bacon will cure it.$
# 01011110 01010011 01101101 01101111 01101011 01101001 01101110 01100111 00100000 01110111 
# 01101001 01101100 01101100 00100000 01101011 01101001 01101100 01101100 00100000 01111001 
# 01101111 01110101 00101100 00100000 01100001 01101110 01100100 00100000 01100010 01100001 
# 01100011 01101111 01101110 00100000 01110111 01101001 01101100 01101100 00100000 01101011 
# 01101001 01101100 01101100 00100000 01111001 01101111 01110101 00101110 00100000 00100000 
# 01000010 01110101 01110100 00101100 00100000 01110011 01101101 01101111 01101011 01101001 
# 01101110 01100111 00100000 01100010 01100001 01100011 01101111 01101110 00100000 01110111 
# 01101001 01101100 01101100 00100000 01100011 01110101 01110010 01100101 00100000 01101001 
# 01110100 00101110 00100100
#############################################################################################
