file = open("data.bin", "r")
interleaved_data = np.fromfile(file, np.uint8)
file.close()

I_data_raw = interleaved_data[0:len(interleaved_data):2]
Q_data_raw = interleaved_data[1:len(interleaved_data):2]
I_samples = (I_data_raw-127.5)/127.5
Q_samples = (Q_data_raw-127.5)/127.5
complex_data = I_samples + 1j*Q_samples

# plt.figure(1)
# plt.plot(abs(complex_data))

ask = complex_data[98845: 98845+11840]
magnitude = np.abs(ask)
d=int(len(ask))
data_range = 2*d