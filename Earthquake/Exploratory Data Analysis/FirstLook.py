
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# take a quick look at some data and simple features


import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join

from scipy import signal, stats

import seaborn as sns
import matplotlib.pyplot as plt


########################################################################################################
# open up training file and see what's here
train = pd.read_csv('train.csv', dtype={ 'acoustic_data': np.int16, 'time_to_failure': np.float64})

print(train.head())
print(train.shape)
print(train.describe())



#########################################################################################################
# Take a section of the training file and plot it
partial_train = train[::200]

figure, axes1 = plt.subplots(figsize=(20, 20))
plt.title('train data')

plt.plot(partial_train['acoustic_data'], color='g')
axes1.set_ylabel('Acoustic data', color='g')

axes2 = axes1.twinx()
plt.plot(partial_train['time_to_failure'], color='r')
axes2.set_ylabel('Time to failure', color='r')

plt.legend()
plt.show()


################################################################################################
# add a few basid statistical features
add_features = ['mean', 'max', 'var', 'min', 'std', 'max-min', 'max-mean',
                'mean-diff-abs', 'abs-max' 'abs-min', 'std-beg', 'std-mid',
                'std-end', 'mean-beg', 'mean-mid', 'mean-end', 'min-beg',
                'min-mid', 'min-end']


# Training files were pre split with Split_Data.py so as to be in same format as testing files
# This took too long and took up too much disk space, final model calculates these on the fly
train_dir = 'train'
train_files = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
n_train = len(train_files)


submission_dir = 'test'
submission_files = [f for f in listdir(submission_dir) if isfile(join(submission_dir, f))]
n_submission = len(submission_files)

features = ['id', 'mean', 'amin', 'amax', 'median', 'std', 'var', 'skew', 'kurt']
targets = ['id', 'time_to_fail']

x = pd.DataFrame(index=range(n_train), columns=features)
y = pd.DataFrame(index=range(n_train), columns=targets)
s = pd.DataFrame(index=range(n_submission), columns=features)


###################################################################
# loop over data files, try a few to make sure everything looks good before running
# through all the files


# training files
for i in range(10):
#for i in range(n_train):

    fname = train_files[i]
    path = 'train/' + fname
    data = pd.read_csv(path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    id = fname.partition('segment_')[2].partition('.cvs')[0]
    
    x_ = data['acoustic_data'].values
    y_ = data['time_to_failure'].values[-1]

    y.loc[i, 'id'] = id
    y.loc[i, 'time_to_fail'] = y_

    x.loc[i, 'id'] = id
    x.loc[i, 'mean'] = x_.mean()
    x.loc[i, 'amin'] = np.amin(x_)

    x.loc[i, 'amax'] = np.amax(x_)
    x.loc[i, 'median'] = np.median(x_)
    x.loc[i, 'std'] = np.std(x_)
    x.loc[i, 'var'] = np.var(x_)

    x.loc[i, 'kurt'] = stats.kurtosis(x_)
    x.loc[i, 'skew'] = stats.skew(x_)
  


# test files    
for i in range(10):
#for i in range(n_submission):

    fname = submission_files[i]
    path = 'test/' + fname
    data = pd.read_csv(path, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    id = fname.partition('seg_')[2].partition('.csv')[0]
    
    x_ = data['acoustic_data'].values

  
    s.loc[i, 'id'] = id
    s.loc[i, 'mean'] = x_.mean()
    s.loc[i, 'amin'] = np.amin(x_)

    s.loc[i, 'amax'] = np.amax(x_)
    s.loc[i, 'median'] = np.median(x_)
    s.loc[i, 'std'] = np.std(x_)
    s.loc[i, 'var'] = np.var(x_)

    s.loc[i, 'kurt'] = stats.kurtosis(x_)
    s.loc[i, 'skew'] = stats.skew(x_)
  



# see how if anything interesting shows up
print(x.head())
print(y.head())
print(s.head())
    
    


