
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



##########################################################################################
# read in a data segment
##########################################################################################

seg = pd.read_csv('train.csv')


print(seg.head())

print(len(seg))



###########################################################################################
# plots
##########################################################################################

fig, ax1 = plt.subplots(figsize=(16,16))
plt.title('Full data !%')

plt.plot(seg['acoustic_data'][::100], color='r')
ax1.set_ylabel('acoustic data', color='r')

ax2 = ax1.twinx()
plt.plot(seg['time_to_failure'][::100], color='b')
ax2.set_ylabel('time to failure', color='b')

plt.grid(True)




plt.show()
