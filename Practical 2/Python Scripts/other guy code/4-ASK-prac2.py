import numpy as np
import cmath
import matplotlib.pyplot as plt

file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()

I_data_raw = interleaved_data[0:len(interleaved_data):2] 
Q_data_raw = interleaved_data[1:len(interleaved_data):2] 
fs = 2.4e6



I_samples = ((I_data_raw-127.5)/127.5)
Q_samples = (Q_data_raw-127.5)/127.5
binsb = 98000
binsf = 125000

y_r = I_samples[binsb:binsf]
raw2 = Q_samples[binsb:binsf]


complex_data = (I_samples + 1j*Q_samples)
complex_data1=complex_data[binsb:binsf]
nbins = len(complex_data1)
t = np.arange(0, nbins/fs, 1/fs)

# freq shift
shifted_complex = complex_data1*(np.cos(2*np.pi*-251496*t*1.846153846)+1j*np.sin(2*np.pi*-251496*t*1.846153846))
plt.figure(10)
plt.plot(I_samples)
plt.xlabel("Time Bins")
plt.ylabel("Normalized Amplitude")
plt.xlim(0,len(I_samples))
plt.title("In-Phase Data (5 Bursts: OOK, 4-ASK, DBPSK, DQPSK, D8PSK)")
plt.show()

c = len(complex_data1)
print(c)
OOK_spliced = shifted_complex*2.718281828**(4j*(np.pi/4)) 
fclock = 55645*1.846153846
print(fclock)
clock =  0.45*np.cos(2*np.pi*fclock*t-0.9)
plt.figure(1)
plt.plot(t,np.abs(OOK_spliced),t,clock)
plt.xlabel("Time Bins")
plt.ylabel("Normalized Magnitude")
plt.grid()
plt.show()


plt.figure(2)
plt.plot(np.abs(OOK_spliced))
plt.xlabel("Time Bins")
plt.ylabel("Normalized Magnitude")
plt.grid()
plt.show()


T_NEW = 866.8184106#183.2637541
a =[]
c= []
b= []
e = 0
for i in np.arange(0,500):
    a.append(OOK_spliced[round(T_NEW)])
    T_NEW = T_NEW + 23.362
for i in np.arange(1,400):
    if(np.angle(a[i])>np.angle(a[i-1])):
        c.append(np.abs(np.angle(a[i])-np.angle(a[i-1])))
    else:
        c.append(2*np.pi-np.abs(np.angle(a[i])-np.angle(a[i-1])))
        
for i in range(len(c)):
    if((np.abs(c[i])<1.5707 and e ==1 ) or (np.abs(c[i]) > 4.18879 and e==1)):
        b.append(cmath.rect(np.abs((a[i])), np.pi))
    elif((np.abs(c[i])<1.5707 and e ==0 ) or (np.abs(c[i]) > 4.18879 and e==0)) :
        b.append(cmath.rect(np.abs((a[i])), 0))
    elif((np.abs(c[i])>1.5707 and e ==1 ) or (np.abs(c[i]) < 4.18879 and e==1)) :
        e=0
        b.append(cmath.rect(np.abs((a[i])), np.pi))
    elif((np.abs(c[i])>1.5707 and e ==0 ) or (np.abs(c[i]) < 4.18879 and e==0)) :
        e = 1 
        b.append(cmath.rect(np.abs((a[i])), 0))
for x in range(len(c)): 
    print(c[x])
plt.figure(3)
plt.plot(np.abs(a))
plt.xlabel("Time Bins")
plt.ylabel("Normalized Magnitude")
plt.grid()
plt.show()


plt.figure(4)
plt.plot(np.real(b), np.imag(b),'o') 
plt.xlabel("In-Phase") 
plt.ylabel("Quadrature") 
plt.grid()
plt.show()

d =[]
bd = ""
count = 0
for i in np.arange(6,300):
    if(count==4):
        d.append(bd)
        bd=""
        count = 0
        if(np.abs(b[i])<=0.3 and np.angle(b[i])==0):
            bd=bd+"11"
            count = count+1
            print(np.angle(b[i]))
        elif(np.abs(b[i])>0.3 and np.angle(b[i])==0 ):
            bd = bd+"10"
            count = count+1
            print(np.angle(b[i]))
        elif(np.abs(b[i])<=(0.3) and np.angle(b[i])==np.pi):
            bd = bd+"01"
            count = count+1
            print(np.angle(b[i]))
        else :
            bd = bd+"00"
            count = count+1
            print(np.angle(b[i]))
    else :
        if(np.abs(b[i])<=0.3 and np.angle(b[i])==0):
            bd=bd+"11"
            count = count+1
            print(np.angle(b[i]))
        elif(np.abs(b[i])>0.3 and np.angle(b[i])==0 ):
            bd = bd+"10"
            count = count+1
            print(np.angle(b[i]))
        elif(np.abs(b[i])<=(0.3) and np.angle(b[i])==np.pi):
            bd = bd+"01"
            count = count+1
            print(np.angle(b[i]))
        else :
            bd = bd+"00"
            count = count+1
            print(np.angle(b[i]))
for x in range(len(d)): 
    print(d[x])
OOK_spliced_further = OOK_spliced[3100:3450]






N_fft = 12000; 

f_1 = np.arange(0, fs, fs/N_fft) 

Y_r = np.fft.fft(y_r, N_fft)/N_fft 



Y_r_mag = 10*np.log10(np.abs(Y_r))  



Y_r_phase = np.angle(Y_r)






f_2 = np.arange(-fs/2, fs/2, fs/N_fft)


Y_r_shifted = np.fft.fftshift(Y_r)


Y_r_shifted_mag = 10*np.log10(np.abs(Y_r_shifted))







Y = np.fft.fft(shifted_complex,N_fft)/N_fft
Y_mag = 10*np.log10(np.abs(Y))





Y_shifted = np.fft.fftshift(Y)
Y_shifted_mag = 10*np.log10(np.abs(Y_shifted))

plt.figure(5)
plt.plot(f_2,Y_shifted_mag)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.xlim(-fs/2, fs/2)
plt.grid()
plt.show()



f_1 = np.arange(0, fs, fs/N_fft) 
Y_r_mag1 = np.abs(complex_data1) 
Y_r = np.fft.fft(Y_r_mag1, N_fft)/N_fft 
Y_r_mag2 = 10*np.log10(Y_r)  





plt.figure(6)
plt.plot(f_1,Y_r_mag2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.xlim(0,fs)
plt.show()

plt.figure(7)
plt.plot(Y_r_mag2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()
plt.show()
Y_r_phase = np.angle(Y_r) 

plt.figure(8)
plt.plot(np.unwrap(Y_r_phase))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid()
plt.show()

OOK_spliced = complex_data[43000:55000]
fclock = 95000
clock =  np.cos(2*np.pi*fclock*t-35) 
