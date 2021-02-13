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

d8psk = complex_data[214620:240220]
d=int(len(d8psk))
magnitude = np.abs(d8psk)


data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(d8psk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))

######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(d8psk ),sample_rate)))

######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(d8psk ))))

######## Removing freq offset #######
f_offset = 284812
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*d8psk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))



###########################
######## Decoding #########
###########################

freq = 94929
angle = 2.22187*0

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = np.cos((2*np.pi*freq*data_x_2+angle))

d8psk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        d8psk_sync.append(centered[i])  


unwrap = np.unwrap(np.angle(d8psk_sync))
# plt.plot(unwrap)

diff_angle = np.diff(unwrap) + (1*np.pi/7)*0
# plt.plot(diff_angle)

sync = np.abs(d8psk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + (1j*(np.sin(diff_angle)))) * np.exp((-8*np.pi/4)*1j)*np.exp((1*np.pi/35)*1j)
unwrap_sync = np.unwrap(np.angle(centered))
# plt.plot(unwrap_sync)


sync_angle = []
sync_mag = []

for j in range(0,int(len(sync))):
    if (np.angle(sync[j]) > -3.12 and np.angle(sync[j]) < -2):
        sync_angle.append(0*np.pi) 
        
    elif (np.angle(sync[j]) > 0.161*np.pi and np.angle(sync[j]) < 0.343*np.pi):
        sync_angle.append(0.25*np.pi) 
    elif (np.angle(sync[j]) > 0.4*np.pi and np.angle(sync[j]) < 0.6*np.pi):
        sync_angle.append(0.5*np.pi) 
    elif (np.angle(sync[j]) > 0.626*np.pi and np.angle(sync[j]) < 0.866*np.pi):
        sync_angle.append(0.75*np.pi)
  
    elif (np.angle(sync[j]) > 0.866*np.pi and np.angle(sync[j]) < 1.12*np.pi):
        sync_angle.append(1*np.pi)
        
    elif (np.angle(sync[j]) > -0.343*np.pi and np.angle(sync[j]) < -0.161*np.pi):
        sync_angle.append(1.25*np.pi)
    elif (np.angle(sync[j]) > -0.6*np.pi and np.angle(sync[j]) < -0.4*np.pi):
        sync_angle.append(1.5*np.pi)
    elif (np.angle(sync[j]) >-1.150  and np.angle(sync[j]) < -0.349 ):
        sync_angle.append(1.75*np.pi)


for k in range(0,int(len(sync_angle))):
    sync_mag.append(1)


decode = []
for j in range(0,int(len(sync))):
    if (np.angle(sync[j]) > -3.15 and np.angle(sync[j]) < -2):
        decode.append("000") 
    elif (np.angle(sync[j]) > 0.161*np.pi and np.angle(sync[j]) < 0.39*np.pi):
        decode.append("001") 
    elif (np.angle(sync[j]) > 0.4*np.pi and np.angle(sync[j]) < 0.62*np.pi):
        decode.append("011") 
    elif (np.angle(sync[j]) > 0.65*np.pi and np.angle(sync[j]) < 0.866*np.pi):
        decode.append("010")
    elif (np.angle(sync[j]) > 0.9*np.pi and np.angle(sync[j]) < 1.12*np.pi):
        decode.append("110")
    elif (np.angle(sync[j]) > -0.343*np.pi and np.angle(sync[j]) < -0.161*np.pi):
        decode.append("111")
    elif (np.angle(sync[j]) > -1.92 and np.angle(sync[j]) < -1.22):
        decode.append("101")
    elif (np.angle(sync[j]) > -1.150  and np.angle(sync[j]) < -0.349 ):
        decode.append("100")


phase = np.angle(sync)
magnitude = np.abs(sync)
soprted = np.sort(phase)
final= np.transpose(decode)
tran = np.transpose(final)

# plt.figure(1)
# plt.plot(unwrap)
# plt.figure(2)
# plt.plot(np.unwrap(np.angle(sync)))

##### Clock vs ABS Data #####
# plt.plot(clock)
# plt.plot(abs(d8psk))
# plt.ylabel("Amlitude")
# plt.xlabel("Time (ms)")

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
# plt.plot(fft_2)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.title("FFT with frequency offset")
# plt.grid()

# plt.figure(2)
# plt.plot(fft_1)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT with frequency offset")
# plt.grid()

##### Constalations #####
# plt.figure(1)
# plt.plot(np.real(d8psk),np.imag(d8psk),"o")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.title("Non Carrier Synced(d8psk)")
# plt.grid()

plt.figure(2)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.title("Carrier Synced(D8PSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()

# plt.figure(3)
# plt.polar(np.angle(sync),np.abs(sync),"o")
# plt.title("Carrier Synced(D8PSK)")