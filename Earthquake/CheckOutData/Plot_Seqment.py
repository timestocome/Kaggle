
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

pd.set_option('display.max_columns', 400)
pd.set_option('display.max_rows', 1000)
pd.set_option('precision', 10)
np.set_printoptions(precision=10)





##########################################################################################
# read in a data segment
##########################################################################################

seg = pd.read_csv('train/segment_35.csv', index_col=0)


print(seg.head(1000))

print(len(seg))



###########################################################################################
# plots
##########################################################################################

'''
# plot full length of data every 100th point
fig, ax1 = plt.subplots(figsize=(16,16))
plt.title('Full Segment, every 100 data')

plt.plot(seg['acoustic_data'].values[::100], color='r')
ax1.set_ylabel('acoustic data', color='r')

ax2 = ax1.twinx()
plt.plot(seg['time_to_failure'].values[::100], color='b')
ax2.set_ylabel('time to failure', color='b')

plt.grid(True)



# plot first 1000 points
fig, ax3 = plt.subplots(figsize=(16,16))
plt.title('Segment first 1000 data points')

plt.plot(seg['acoustic_data'].values[:1000], color='r')
ax3.set_ylabel('acoustic data', color='r')

ax4 = ax3.twinx()
plt.plot(seg['time_to_failure'].values[:1000], color='b')
ax4.set_ylabel('time to failure', color='b')

plt.grid(True)
'''


total_counts = seg['acoustic_data'].value_counts()
density = stats.kde.gaussian_kde(total_counts)

print (density)



#plt.show()
