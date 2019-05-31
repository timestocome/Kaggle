

# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

# http://scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html

# compute FFT on data to get power to use as features


import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import pandas as pd


##########################################################################################
# read in a data segment
##########################################################################################

seg = pd.read_csv('train/segment_4136.csv', index_col=0)

features = ['acoustic_data']
targets = ['time_to_failure']

#print(seg.head())
#print(seg.describe())

n_samples = len(seg)


#############################################################################################
# FFT stuff
###########################################################################################

signal = seg['acoustic_data']


plt.figure(figsize=(12,12))
plt.plot(signal, label='Acoustic data')
plt.show()



#-----------------------------------------------------------
hz = 40000
period = 3.75
time = np.arange(0, n_samples)

# compute power
sig_fft = fftpack.fft(signal)
power = np.abs(sig_fft)
sample_freq = fftpack.fftfreq(signal.size, d=hz)

plt.figure(figsize=(12,12))
plt.plot(sample_freq, power)
plt.xlabel('Frequency Hz')
plt.ylabel('Power')

plt.show()


#-------------------------------------------------------------------

# find peak frequency
# remove frequencies less than zero
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]

# get location in power array of largest power 
peak_freq = freqs[power[pos_mask].argmax()]

np.allclose(peak_freq, 1./period)



#----------------------------

#axes = plt.axes([0.0, 15.0, 0.0, 15.0])
plt.figure(figsize=(16,16))
plt.title('peak frequency')
plt.plot(freqs, power[:74999])
#plt.setp(axes, yticks=[])
plt.show()

print('peak frequency')
print('freq', len(freqs), freqs[:10] * hz)
print('power', len(power), power[:10])



# ----------------------------------------
# remove high freqs

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

print('filtered sig', len(filtered_sig))

plt.figure(figsize=(12,12))
plt.plot(time, signal, label='Original', c='b')
plt.plot(time, filtered_sig, label='Filtered', c='r')
plt.xlabel('time')
plt.ylabel('amplitude')

plt.legend(loc='best')
plt.show()
