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

msk = complex_data[245255:269502]

# plt.figure(1)
# plt.plot(abs(complex_data))
# plt.grid()


d=int(len(msk))

# plt.figure(1)
# plt.plot(abs(msk))

magnitude = np.abs(msk)
data_range = 2*d

######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(msk ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(msk ),sample_rate)))
#plt.plot(fft_1)

######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(msk))))





######## Removing freq offset #######
f_offset = 478800-5500
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*msk


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))




freq = 100000-1
angle = -0.68*np.pi/2

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = 1*np.cos(2*np.pi*freq*clock_range+angle)

msk_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        msk_sync.append(centered[i])  


unwrap = np.unwrap(np.angle(msk_sync))
diff_angle = np.diff(unwrap) 
sync = np.abs(msk_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 1j*(np.sin(diff_angle))) *np.exp((2*np.pi/2)*1j)


points =[]
for j in range(0,int(len(sync))):
     
    if (np.imag(sync[j]) > 1j* 0.0):
        points.append("0")
    elif (np.imag(sync[j]) < 1j* 0.0):
        points.append("1")

final = np.transpose(points[2:688]) #data bit to convert




############ Ploting ####################

###### Plots FFTs and PSDs #########
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
# plt.plot(abs(fft_1),'b')
# # plt.xlim(-20000,sample_rate)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("Frequency (Hz)")
# plt.title("FFT Of MSK")
# plt.grid()



# plt.plot(np.unwrap(np.angle(magnitude)))
# plt.plot(np.unwrap(np.angle(msk_sync)))










# ### Clocks
# plt.figure(4)
# plt.plot(clock,'b')
# plt.plot(np.abs(msk),'r')
# plt.title("Clock vs |MSK|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()

# plt.figure(1)
# plt.plot(np.real(msk)/80,np.imag(msk)/75,"o")
# plt.title(" Non Carrier Synced(MSK)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


plt.figure(2)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.title("Carrier Synced(MSK)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


# plt.figure(3)
# plt.polar(np.angle(sync[0:1010]),np.abs(sync[0:1010]),"o")
# plt.title("Carrier Synced(MSK)")
# plt.plot()

plt.show()


############################################################################################
# ^The dinner I was cooking was going to be delicious, but the fire engine ruined it.$
# 01011110 01010100 01101000 01100101 00100000 01100100 01101001 01101110 01101110 01100101 
# 01110010 00100000 01001001 00100000 01110111 01100001 01110011 00100000 01100011 01101111 
# 01101111 01101011 01101001 01101110 01100111 00100000 01110111 01100001 01110011 00100000 
# 01100111 01101111 01101001 01101110 01100111 00100000 01110100 01101111 00100000 01100010 
# 01100101 00100000 01100100 01100101 01101100 01101001 01100011 01101001 01101111 01110101 
# 01110011 00101100 00100000 01100010 01110101 01110100 00100000 01110100 01101000 01100101 
# 00100000 01100111 01101001 01110010 01100101 00100000 01100101 01101110 01100111 01101001 
# 01101110 01100101 00100000 01110010 01110101 01101001 01101110 01100101 01100100 00100000 
# 01101001 01110100 00101110 00100100
###########################################################################################