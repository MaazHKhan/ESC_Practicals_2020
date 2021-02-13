######################################
#### Resampling
#####################################

freq = 100000
angle = np.pi

data_x_2 = np.linspace(0, d/sample_rate, d)
clock =0.45*np.cos((2*np.pi*freq*data_x_2+angle-0.5))

huffman_sync = []
for i in range (d-27820,d-1):
    if (clock[i] > clock[i-1] and clock[i] > clock[i+1]):
        huffman_sync.append(huffman[i])  


unwrap = np.unwrap(np.angle(huffman_sync))
diff_angle = np.diff(unwrap)
sync = np.abs(huffman_sync[0:int(len(diff_angle))]) * (np.cos(diff_angle) + 0*1j*(np.sin(diff_angle)*1)) # -np.exp((1*np.pi/4.5)*1j)


decode = []
for j in range(0,int(len(sync))):
    if (sync[j] > 0):
        decode.append("0")   
    elif (sync[j] < 0):
        decode.append("1")
        
samooosa = []
for i in range(0 ,len(decode),5): 
    samooosa.append(decode[i:i+5])
