# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:29:26 2020

@author: maazkhan
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft, ifft



###### Open the binary file and read to a variable ######
file = open("data.bin","rb")
byte = file.read(4000000)
length = len(byte)
x = int(length*0.5)


#### takes the bianary read data and converts it to conplex values ####
data_list= [] 
for i in range (0,x): 
    data_list.append(byte[2*i]+1j*byte[2*i+1])
    
#### remove the offest and cerntre data at 0+0j (normalised) ####
center_data = []
for j in range (0,x): 
    center_data.append(data_list[j]-125-125*1j)


#### Spliting up the messages ####

OOK = center_data[14611:37897]
ASK = center_data[43085:68767]
DBPSK = center_data[74000:99250]
DQPSK = center_data[104501:128760]
D8PSK = center_data[136026:159580]



### Saving to textfile ###

# np.savetxt("ook.txt",OOK)
# np.savetxt("ask.txt",ASK)
# np.savetxt("dbpsk.txt",DBPSK)
# np.savetxt("dqpsk.txt",DQPSK)
np.savetxt("d8psk.txt",D8PSK)



#### PLOTS ####  

# plt.subplot(2,1,1)
# plt.title('Raw Data')
# plt.plot(data_list,'g')
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.grid()

# plt.subplot(2,1,2)
# plt.plot(center_data, 'r')
# plt.ylabel('Amplitude')
# plt.xlabel('Time')
# plt.title('Normalised Data')
# plt.grid()



# plt.subplots_adjust(hspace=1)
# plt.rc('font', size=15)
# fig = plt.gcf()
# fig.set_size_inches(20, 12)






plt.plot(np.real(data_list),np.imag(data_list),"o") # non normalised data
plt.plot(np.real(center_data),np.imag(center_data),"o") # normalised data




plt.show()