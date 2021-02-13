import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath

sample_rate = 2400000
rate = 2400000

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

ook = complex_data[68000:93600]

d=len(ook)
magnitude = np.abs(ook)


data_range = 2*d

######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(ook ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))

######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(ook ),sample_rate)))

######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(ook ))))

######## Removing freq offset #######
f_offset = -650000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*ook


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))

fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))


########## Down Samplimg ########
freq = 94930
angle = 0*(np.pi)/2


data_x_2 = np.linspace(0, d/rate, d)
clock = 1*np.cos((2*np.pi*freq*data_x_2+angle))


### clock Sync ###
ook_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        ook_sync.append(ook[i])   


### Decode ####
use = np.abs(ook_sync)
recovery = []
for j in range (0,1010):
        if ((use[j] > 0.2)):
            recovery.append("1")
        else:
            recovery.append("0")
        
unwrap = np.unwrap(np.angle(ook_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(ook_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) *np.exp((-1*np.pi/59)*1j) * np.exp((-1*0*np.pi/2)*1j)


decode = np.transpose(recovery[0:1010])  
 

 
# print(np.angle((ook[1011])))

polar = []

recovery = []
for j in range (0,1010):
        if ((use[j] > 0.2)):
            polar.append(1)
        else:
            polar.append(0)

############ Ploting ####################

#### Clock vs ABS Data #####
# plt.plot(clock,'b')
# plt.plot(magnitude,'r')
# plt.title("Clock vs |OOK|")
# plt.ylabel("Amlitude")
# plt.xlabel("Time ")

# plt.figure(1)
# plt.plot(data_x,psd,'r')
# plt.plot(data_x,psd_offset,'b')
# # plt.ylim(0,80)
# # plt.xlim(-200000,150000)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("Power Spectral Density")
# plt.grid()

# plt.figure(2)
# plt.plot(fft_1,'b')
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT Of DBPSK")
# plt.grid()

#### Constalations #####
# plt.figure(3)
# plt.plot(np.real(ook),np.imag(ook),"o")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.title("Non Carrier Synced(OOK)")
# plt.grid()

plt.figure(4)
plt.plot(np.real(ook_sync),np.imag(ook_sync),"o")
plt.title("Carrier Synced(OOK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()



# plt.figure(5)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.title("Carrier Synced(OOK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()

plt.figure(5)
plt.polar(np.angle(polar),np.abs(polar ),"o")
plt.title("Carrier Synced(OOK)")

plt.show()


##########################################################################
# ^Atheism is a non-prophet organization.$
# 01011110 01000001 01110100 01101000 01100101 01101001 01110011 01101101 
# 00100000 01101001 01110011 00100000 01100001 00100000 01101110 01101111 
# 01101110 00101101 01110000 01110010 01101111 01110000 01101000 01100101 
# 01110100 00100000 01101111 01110010 01100111 01100001 01101110 01101001 
# 01111010 01100001 01110100 01101001 01101111 01101110 00101110 00100100
##########################################################################