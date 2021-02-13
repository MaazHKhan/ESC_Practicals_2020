# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 22:14:30 2020

@author: maazk
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

block = complex_data[73656:99700]

plt.figure(1)
plt.plot(abs(block))
plt.grid()


d=int(len(block))

# plt.figure(1)
# plt.plot(abs(block))

magnitude = np.abs(block)

data_range = 2*d
######### code for PSD plots ############
data_x = np.arange(-sample_rate/2 ,sample_rate/2 ,sample_rate/data_range)

freq_range = np.arange(0,sample_rate, sample_rate/data_range)
fft = np.fft.fft(block ,data_range)
psd_shift = np.fft.fftshift(fft)
psd = abs(20*np.log10(np.abs(psd_shift)))


######## Code for FFT Plots #########
fft_1  = 20*np.log10(np.abs(np.fft.fft(np.abs(block ),sample_rate)))


######## Code for fft bins  #########
fft_2  = 20*np.log10(np.abs(np.fft.fft(np.abs(block ))))






######################################
############# Resampling ############# 
######################################

freq = 100000
angle = 2*0

data_x_2 = np.linspace(0, d/sample_rate, d)
clock = np.cos((2*np.pi*freq*data_x_2+angle))

block_sync = []
for i in range (0,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        block_sync.append(block[i])  


unwrap = np.unwrap(np.angle(block_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(block_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 0*1j*(np.sin(diff_angle)*1))


decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0):
        decode.append("0")   
    elif (sync[j] < 0):
        decode.append("1")


final = np.transpose(decode)


final_x = []
for i in range (1086, 1489):
        final_x.append("0")

for i in range(0,int(len(final))):
    final_x.append(final[i])


final_xx = np.int32(np.transpose(final_x))



data = final_xx #np.genfromtxt("blockdata.txt",dtype=int)

g = np.array([[1,0,0,0,1,1],[0,1,0,1,0,1],[0,0,1,1,1,1]]) #Genarator
h = np.array([[0,1,1,1,0,0],[1,0,1,0,1,0],[1,1,1,0,0,1]]) #Parity check matrix
ht = np.transpose(h)                                      #Transpose of parity check matrix
print("Genarator =")
print(g)
print("")
print("Paraty check matrix =")
print(h)
print("")
print("Transpose of Paraty check matrix =")
print(ht)


dataoutput = np.zeros(800,dtype = int)
for i in range(200):
    testdata = np.array([data[6*i],data[(6*i)+1],data[(6*i)+2],data[(6*i)+3],data[(6*i)+4],data[(6*i)+5]])
    print(testdata)

    syndrome = (np.dot(testdata,ht))%2
    
    outdata = np.array([data[6*i],data[(6*i)+1],data[(6*i)+2]])
    print(outdata)
    print(syndrome)
    left = np.array([1,0,0])
    mid = np.array([0,1,0])
    right = np.array([0,0,1])
    leftcheck = np.array([0,1,1])
    midcheck = np.array([1,0,1])
    rightcheck = np.array([1,1,1])
    
    if (syndrome[0]==0 and syndrome[1]==1 and syndrome[2]==1):
        finaldata = outdata ^ left
    elif(syndrome[0]==1 and syndrome[1]==0 and syndrome[2]==1):
        finaldata = outdata ^ mid
    elif(syndrome[0]==1 and syndrome[1]==1 and syndrome[2]==1):
        finaldata = outdata ^ right
    else:
        finaldata = outdata
    print(finaldata)
    dataoutput[i*3] = finaldata[0]
    dataoutput[(i*3)+1] = finaldata[1]
    dataoutput[(i*3)+2] = finaldata[2]
  
print(dataoutput)














# plt.figure(8)
# plt.plot(np.real(sync),np.imag(sync),"o")
# plt.ylim(-1,1)
# plt.title("Carrier Synced(huffman)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()


# plt.figure(9)
# plt.plot(np.real(block_sync),np.imag(block_sync),"o")
# plt.ylim(-1,1)
# plt.title("Carrier Synced(huffman)")
# plt.ylabel("In-Quadrature")
# plt.xlabel("In-Phase")
# plt.grid()




