
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# plot acoustic data against target and see if anything jumps out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



##########################################################################################
# read in a data segment or the whole training file
##########################################################################################

data = pd.read_csv('train.csv')
print(data.describe())

print(data.head())
print(len(data))


acoustic_mean = 4.519468
cap = 150.



data['acoustic_data'] = data['acoustic_data'] - acoustic_mean
data['acoustic_data'] = np.clip(data['acoustic_data'], -cap, cap)

data['acoustic_data'] = np.abs(data['acoustic_data'])


###########################################################################################
# plot data but only every 100th point to keep from overwheling computer
##########################################################################################

fig, ax1 = plt.subplots(figsize=(32,16))
plt.title('Full data 1%')

plt.plot(data['acoustic_data'][::1000], color='r')
ax1.set_ylabel('acoustic data', color='r')
#ax1.set_ylim(-1000, 1000)

ax2 = ax1.twinx()
plt.plot(data['time_to_failure'][::1000], color='b')
ax2.set_ylabel('time to failure', color='b')

plt.grid(True)

plt.savefig('PlotQuakes.png')
plt.show()




