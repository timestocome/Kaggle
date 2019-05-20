
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy import stats
from scipy import ndimage as ndi


pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 100)
pd.set_option('precision', 10)
np.set_printoptions(precision=10)





##########################################################################################
# read in a data segment
##########################################################################################

seg = pd.read_csv('train/segment_4136.csv', index_col=0)

features = ['acoustic_data']
targets = ['time_to_failure']

#print(seg.head())
#print(seg.describe())

n_samples = len(seg)
#print(len(seg))

#############################################################################################
# add some basic stats, nothing fancy
###########################################################################################

'''

window = 150000 - 50
print('window', window)


seg['min'] = seg['acoustic_data'].rolling(window=65536).min()
seg['max'] = seg['acoustic_data'].rolling(window=65536).max()
seg['median'] = seg['acoustic_data'].rolling(window=8192).median()
seg['skew'] = seg['acoustic_data'].rolling(window=window).skew()
seg['kurt'] = seg['acoustic_data'].rolling(window=window).kurt()
seg['sum'] = seg['acoustic_data'].rolling(window=window).sum()
seg['var'] = seg['acoustic_data'].rolling(window=window).var()
seg['std'] = seg['acoustic_data'].rolling(window=window).std()


print(np.abs(seg.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])
'''



seg['diff1'] = seg['acoustic_data'] - seg['acoustic_data'].shift(1)
seg['diff2'] = seg['diff1'] - seg['diff1'].shift(1)
print(seg.head())
seg.fillna(0., inplace=True)
'''
fig, ax1 = plt.subplots(figsize=(16,16))
plt.title('Full data !%')

plt.plot(seg['acoustic_data'], color='r')
plt.plot(seg['diff1'], color='g')
plt.plot(seg['diff2'], color='y', alpha=.2)
ax1.set_ylabel('acoustic data', color='r')

ax2 = ax1.twinx()
plt.plot(seg['time_to_failure'], color='b')
ax2.set_ylabel('time to failure', color='b')

plt.grid(True)
'''

filter = np.array([1, 0, -1])
dsig = ndi.convolve(seg['acoustic_data'], filter)

fig, ax = plt.subplots(figsize=(16,16))
plt.plot(seg['acoustic_data'], c='r')
plt.plot(dsig, alpha=0.5, c='b')



plt.show()
