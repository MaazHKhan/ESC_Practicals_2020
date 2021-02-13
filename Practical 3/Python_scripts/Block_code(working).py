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
decode.append("0")
for j in range(0,int(len(sync))):
    if (sync[j] > 0.1):
        decode.append("0")   
    elif (sync[j] < -0.1):
        decode.append("1")


final = np.transpose(decode)






data = np.genfromtxt("blockdata.txt",dtype=int) #final_xx #

g = np.array([[1,0,0,0,1,1],[0,1,0,1,0,1],[0,0,1,1,1,1]]) #Genarator
h = np.array([[0,1,1,1,0,0],[1,0,1,0,1,0],[1,1,1,0,0,1]]) #Parity check matrix
ht = np.transpose(h)     
print("")                                 #Transpose of parity check matrix
print("Genarator =")
print(g)
print("")
print("Paraty check matrix =")
print(h)
print("")
print("Transpose of Paraty check matrix =")
print(ht)


data_r = np.reshape(data,(181,6))

syndrome = (np.dot(data_r,ht))%2

print("Syndrome for the first 10 recived code words:\n",syndrome[1:10])

error_1 = [0,0,0,0,0,1]
error_2 = [0,0,0,0,1,0]
error_3 = [0,0,0,1,0,0]
error_4 = [0,1,0,0,0,0]
error_5 = [0,0,1,0,0,0]
error_6 = [0,0,0,0,0,0]
error_7 = [1,0,0,0,1,0]
error_8 = [1,0,0,0,0,0]

error = []

for i in range(0,181):
    if (syndrome[i,0]==0 and syndrome[i,1]==0 and syndrome[i,2]==1):
        error.append(error_1)
    elif(syndrome[i,0]==0 and syndrome[i,1]==1 and syndrome[i,2]==0):
        error.append(error_2)
    elif(syndrome[i,0]==1 and syndrome[i,1]==0 and syndrome[i,2]==0):
        error.append(error_3)
    elif(syndrome[i,0]==1 and syndrome[i,1]==0 and syndrome[i,2]==1):
        error.append(error_4)
    elif(syndrome[i,0]==1 and syndrome[i,1]==1 and syndrome[i,2]==1):
        error.append(error_5)
    elif(syndrome[i,0]==0 and syndrome[i,1]==0 and syndrome[i,2]==0):
        error.append(error_6)
    elif(syndrome[i,0]==1 and syndrome[i,1]==1 and syndrome[i,2]==0):
        error.append(error_7)
    elif(syndrome[i,0]==0 and syndrome[i,1]==1 and syndrome[i,2]==1):
        error.append(error_8)

code_word = (error + data_r)%2

data_out = []

for i in range(0,len(code_word)):
    data_out.append(code_word[i,0:3])

out = np.transpose(data_out)
out_also = np.transpose(out)

result = out.flatten('F')

final_result = result[3:543]

print("")
print("Data to decode:")
print(final_result)
   



plt.figure(8)
plt.plot(np.real(sync),np.imag(sync),"o")
plt.ylim(-1,1)
plt.title("Carrier Synced(huffman)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()


plt.figure(9)
plt.plot(np.real(block_sync),np.imag(block_sync),"o")
plt.ylim(-1,1)
plt.title("Carrier Synced(huffman)")
plt.ylabel("In-Quadrature")
plt.xlabel("In-Phase")
plt.grid()




