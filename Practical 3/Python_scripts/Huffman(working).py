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

huffman = complex_data[44000:68400]

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


######## Removing freq offset #######
f_offset = 0
# f_offset = 342365
clock_range = np.linspace(0,d/sample_rate,d)
clock = 1*(np.cos(2*np.pi*f_offset*clock_range)+1j*np.sin(2*np.pi*f_offset*clock_range))
centered = clock*huffman


######## Code for PSD centered  #########
fft3 = np.fft.fft(centered,data_range)
psd_shift2 = np.fft.fftshift(fft3)
psd_offset = abs(20*np.log10(np.abs(psd_shift2)))


fft_4 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered),sample_rate)))
fft_5 = 20*np.log10(np.abs(np.fft.fft(np.abs(centered))))


######################################
#### Resampling
#####################################

freq = 100000
angle = 2

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = np.cos((2*np.pi*freq*data_x_2+angle))

huffman_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        huffman_sync.append(centered[i])  


unwrap = np.unwrap(np.angle(huffman_sync))

diff_angle = np.diff(unwrap)

sync = np.abs(huffman_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 0*1j*(np.sin(diff_angle)*1)) #* -np.exp((1*np.pi/4.5)*1j)



decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0):
        decode.append("0")   
    elif (sync[j] < 0):
        decode.append("1")


final = np.transpose(decode)
np.savetxt("demod.txt",decode,fmt='%s')


##############################################################################
############################## Huffman Decoding ##############################
##############################################################################
b=12
demod=[]
bits=[]
huffman = []
demod = np.genfromtxt("demod.txt")
print(demod)
plt.show() 
j=0


for j in range (len(demod)):
    if demod[j] == 0:
        bits.append(0)
        j=j+1
    elif demod[j] == 1:
        bits.append(1)
        j=j+1
    

huff = np.chararray(100,unicode='true')

for i in range (0,96):
    
    if bits[b:b+3] == [0,0,0]:
        huff[i]=(" ")
        b = b+3
        
    elif bits[b:b+7] == [0,0,1,0,0,0,0]:
        huff[i]=(".")
        b=b+7
        
    elif bits[b:b+8] == [0,0,1,0,0,0,1,0]:
        huff[i]=("$")
        b=b+8
    
    elif bits[b:b+8] == [0,0,1,0,0,0,1,1]:
        huff[i]=("^")
        b=b+8
        
    elif bits[b:b+8] == [0,0,1,0,1,1,0,0]:
        huff[i]=(",")
        b=b+8
        
    elif bits[b:b+8] == [0,0,1,0,1,1,0,1]:
        huff[i]=("'")
        b=b+8
        
    elif bits[b:b+4] == [0,1,1,0]:
        huff[i]=("a")
        b=b+4
        
    elif bits[b:b+6] == [1,0,0,0,1,1]:
        huff[i]=("b")
        b=b+6
        
    elif bits[b:b+6] == [0,0,1,0,0,1]:
        huff[i]=("c")
        b=b+6
        
    elif bits[b:b+5] == [0,1,0,0,1]:
        huff[i]=("d")
        b=b+5
        
    elif bits[b:b+3] == [1,1,0]:
        huff[i]=("e")
        b=b+3
        
    elif bits[b:b+6] == [0,1,0,1,0,0]:
        huff[i]=("f")
        b=b+6
        
    elif bits[b:b+6] == [0,1,0,1,0,1]:
        huff[i]=("g")
        b=b+6
         
    elif bits[b:b+4] == [1,1,1,0]:
        huff[i]=("h")
        b=b+4
        
    elif bits[b:b+4] == [1,0,0,1]:
        huff[i]=("i")
        b=b+4
        
    elif bits[b:b+9] == [1,0,0,0,1,0,0,1,0]:
        huff[i]=("j")
        b=b+9
        
    elif bits[b:b+7] == [1,0,0,0,1,0,1]:
        huff[i]=("k")
        b=b+7
        
    elif bits[b:b+5] == [0,1,0,1,1]:
        huff[i]=("l")
        b=b+5
        
    elif bits[b:b+6] == [0,1,0,0,0,0]:
        huff[i]=("m")
        b=b+6
        
    elif bits[b:b+4] == [1,0,1,0]:
        huff[i]=("n")
        b=b+4
        
    elif bits[b:b+4] == [0,1,1,1]:
        huff[i]=("o")
        b=b+4
        
    elif bits[b:b+6] == [1,0,0,0,0,1]:
        huff[i]=("p")
        b=b+6
        
    elif bits[b:b+10] == [1,0,0,0,1,0,0,0,1,1]:
        huff[i]=("q")
        b=b+10
        
    elif bits[b:b+4] == [1,1,1,1]:
        huff[i]=("r")
        b=b+4
        
    elif bits[b:b+4] == [1,0,1,1]:
        huff[i]=("s")
        b=b+4
        
    elif bits[b:b+4] == [0,0,1,1]:
        huff[i]=("t")
        b=b+4
        
    elif bits[b:b+6] == [0,0,1,0,1,0]:
        huff[i]=("u")
        b=b+6
        
    elif bits[b:b+7] == [0,0,1,0,1,1,1]:
        huff[i]=("v")
        b=b+7
        
    elif bits[b:b+6] == [0,1,0,0,0,1]:
        huff[i]=("w")
        b=b+6
        
    elif bits[b:b+9] == [1,0,0,0,1,0,0,1,1]:
        huff[i]=("x")
        b=b+9
        
    elif bits[b:b+6] == [1,0,0,0,0,0]:
        huff[i]=("y")
        b=b+6
        
    elif bits[b:b+11] == [1,0,0,0,1,0,0,0,1,0,1]:
        huff[i]=("z")
        b=b+11
        
    elif bits[b:b+11] == [1,0,0,0,1,0,0,0,1,0,0]:
        huff[i]=("?")
        b=b+11
        
    else:
        huff[i]=('%')
        b=b+1


final_huff = np.transpose(huff)
print("The joke is:\n")
print(final_huff.strip(' '))


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
# plt.plot(fft_1)
# plt.ylabel("Magnitude (dB)")
# plt.xlabel("samples")
# plt.title("FFT with frequency offset")
# plt.grid()


# plt.figure(4)
# plt.plot(np.angle(huffman ))
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
# plt.plot(np.abs(huffman),'r')
# plt.title("Clock vs |huffman|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()


##### Constatlations
# plt.figure(7)
# plt.polar(np.angle(sync[0:len(points)]),np.abs(points),"o")
# plt.title("Carrier Synced(huffman)")

plt.figure(8)
plt.plot(np.real(huffman_sync),np.imag(huffman_sync),"o")
plt.ylim(-1,1)
plt.title("Carrier Synced(huffman)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


plt.figure(9)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.ylim(-1,1)
plt.title("Carrier Synced(huffman)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()



# plt.figure(9)
# plt.plot(np.real(huffman),np.imag(huffman),"o")
# plt.title("Non Carrier Synced(huffman)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()



# ### Clocks
# plt.figure(3)
# plt.plot(clock,'b')
# plt.plot(np.abs(huffman),'r')
# plt.title("Clock vs |huffman|")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.grid()

