import numpy as np
from matplotlib import pyplot as plt
from pylab import *
# import cmath

sample_rate = 2400000
rate = 2400000
#######################################################################
########################### Reading in data ########################### 
#######################################################################
file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()


I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples
#######################################################################
#######################################################################

dqpsk = complex_data[185730:209400]
d=int(len(dqpsk))
magnitude = np.abs(dqpsk)

# plt.plot(complex_data)



data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(dqpsk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(dqpsk ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(dqpsk ))))


######## Removing freq offset #######
f_offset = 100000
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*dqpsk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))




freq = 102732
angle = 1.73*2
data_x_2 = np.linspace(0,d/sample_rate,d)
clock = 1*np.cos(2*np.pi*freq*data_x_2+angle)

dqpsk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        dqpsk_sync.append(centered[i])  

unwrap = np.unwrap(np.angle(dqpsk_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(dqpsk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) *np.exp((-1*np.pi/59)*1j) * np.exp((-1*0*np.pi/2)*1j)

points =[]
for j in range(0,int(len(sync))):
    if (sync[j] > 0.25):
        points.append("00")   
    elif (sync[j] < -0.25):
        points.append("11")     
    elif (np.imag(sync[j]) > 1j* 0.25):
        points.append("01")
    elif (np.imag(sync[j]) < -1j* 0.25):
        points.append("10")

final = np.transpose(points[3:384])




############ Ploting ####################

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

# plt.figure(2)
# plt.plot(fft_1,'b')
# plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT Of DQPSK")
# plt.grid()

# # ### Clocks
# plt.figure(3)
# plt.plot(clock,'b')
# plt.plot(np.abs(dqpsk),'r')
# plt.title("Clock vs |DQPSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()

# plt.figure(4)
# plt.plot(np.real(dqpsk),np.imag(dqpsk),"o")
# plt.title(" Non Carrier Synced(DQSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


plt.figure(5)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.title("Non Carrier Synced(DBPSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


# plt.figure(6)
# plt.polar(np.angle(sync[0:384]),np.abs(sync[0:384]),"o")
# plt.title("Carrier Synced(DQSK)")
# plt.plot()

plt.show()


#####################################################################################################
# ^I bought a vacuum cleaner six months ago,and so, far all it's been doing is gathering dust.$
# 01011110 01001001 00100000 01100010 01101111 01110101 01100111 01101000 01110100 00100000 01100001 
# 00100000 01110110 01100001 01100011 01110101 01110101 01101101 00100000 01100011 01101100 01100101 
# 01100001 01101110 01100101 01110010 00100000 01110011 01101001 01111000 00100000 01101101 01101111 
# 01101110 01110100 01101000 01110011 00100000 01100001 01100111 01101111 00101100 01100001 01101110 
# 01100100 00100000 01110011 01101111 00101100 00100000 01100110 01100001 01110010 00100000 01100001 
# 01101100 01101100 00100000 01101001 01110100 00100111 01110011 00100000 01100010 01100101 01100101 
# 01101110 00100000 01100100 01101111 01101001 01101110 01100111 00100000 01101001 01110011 00100000 
# 01100111 01100001 01110100 01101000 01100101 01110010 01101001 01101110 01100111 00100000 01100100 
# 01110101 01110011 01110100 00101110 00100100
#####################################################################################################