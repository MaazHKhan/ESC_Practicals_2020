import csv
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import cmath
from scipy.signal import butter, lfilter, freqz, filtfilt


b=12
demod=[]
value=[]
huffman = []
demod = np.genfromtxt("demod.txt")
print(demod)
plt.show() 
j=0



for j in range (len(demod)):
    if demod[j] == 0:
        value.append(0)
        j=j+1
    elif demod[j] == 1:
        value.append(1)
        j=j+1
    
        

for i in range (0,97):
    
    if value[b:b+3] == [0,0,0]:
        huffman.append(" ")
        b = b+3
        
    elif value[b:b+7] == [0,0,1,0,0,0,0]:
        huffman.append(".")
        b=b+7
        
    elif value[b:b+8] == [0,0,1,0,0,0,1,0]:
        huffman.append("$")
        b=b+8
        
    elif value[b:b+8] == [0,0,1,0,0,0,1,1]:
        huffman.append("^")
        b=b+8
        
    elif value[b:b+8] == [0,0,1,0,1,1,0,0]:
        huffman.append(",")
        b=b+8
        
    elif value[b:b+8] == [0,0,1,0,1,1,0,1]:
        huffman.append("'")
        b=b+8
        
    elif value[b:b+4] == [0,1,1,0]:
        huffman.append("a")
        b=b+4
        
    elif value[b:b+6] == [1,0,0,0,1,1]:
        huffman.append("b")
        b=b+6
        
    elif value[b:b+6] == [0,0,1,0,0,1]:
        huffman.append("c")
        b=b+6
        
    elif value[b:b+5] == [0,1,0,0,1]:
        huffman.append("d")
        b=b+5
        
    elif value[b:b+3] == [1,1,0]:
        huffman.append("e")
        b=b+3
        
    elif value[b:b+6] == [0,1,0,1,0,0]:
        huffman.append("f")
        b=b+6
        
    elif value[b:b+6] == [0,1,0,1,0,1]:
        huffman.append("g")
        b=b+6
         
    elif value[b:b+4] == [1,1,1,0]:
        huffman.append("h")
        b=b+4
        
    elif value[b:b+4] == [1,0,0,1]:
        huffman.append("i")
        b=b+4
        
    elif value[b:b+9] == [1,0,0,0,1,0,0,1,0]:
        huffman.append("j")
        b=b+9
        
    elif value[b:b+7] == [1,0,0,0,1,0,1]:
        huffman.append("k")
        b=b+7
        
    elif value[b:b+5] == [0,1,0,1,1]:
        huffman.append("l")
        b=b+5
        
    elif value[b:b+6] == [0,1,0,0,0,0]:
        huffman.append("m")
        b=b+6
        
    elif value[b:b+4] == [1,0,1,0]:
        huffman.append("n")
        b=b+4
        
    elif value[b:b+4] == [0,1,1,1]:
        huffman.append("o")
        b=b+4
        
    elif value[b:b+6] == [1,0,0,0,0,1]:
        huffman.append("p")
        b=b+6
        
    elif value[b:b+10] == [1,0,0,0,1,0,0,0,1,1]:
        huffman.append("q")
        b=b+10
        
    elif value[b:b+4] == [1,1,1,1]:
        huffman.append("r")
        b=b+4
        
    elif value[b:b+4] == [1,0,1,1]:
        huffman.append("s")
        b=b+4
        
    elif value[b:b+4] == [0,0,1,1]:
        huffman.append("t")
        b=b+4
        
    elif value[b:b+6] == [0,0,1,0,1,0]:
        huffman.append("u")
        b=b+6
        
    elif value[b:b+7] == [0,0,1,0,1,1,1]:
        huffman.append("v")
        b=b+7
        
    elif value[b:b+6] == [0,1,0,0,0,1]:
        huffman.append("w")
        b=b+6
        
    elif value[b:b+9] == [1,0,0,0,1,0,0,1,1]:
        huffman.append("x")
        b=b+9
        
    elif value[b:b+6] == [1,0,0,0,0,0]:
        huffman.append("y")
        b=b+6
        
    elif value[b:b+11] == [1,0,0,0,1,0,0,0,1,0,1]:
        huffman.append("z")
        b=b+11
        
    elif value[b:b+11] == [1,0,0,0,1,0,0,0,1,0,0]:
        huffman.append("?")
        b=b+11
        
    else:
        huffman.append("%")
        b=b+1

        
       
print(len(huffman))  
print(len(value))  
print(len(demod))
print(huffman)

        
        
        


        
    
    
       
        
        
        
        


        
    
    