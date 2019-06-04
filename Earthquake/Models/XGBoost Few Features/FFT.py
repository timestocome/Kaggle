
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# iterative program - create features build train and test 
# dataframes with new features

##############################################################
# libraries used
################################################################

import numpy as np
import scipy.stats as stats
import scipy.fftpack
import scipy.signal
import pandas as pd
import os
import matplotlib.pyplot as plt



###############################################################
# print options
##############################################################
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.float_format', '{:15.11f}'.format)

np.set_printoptions(precision=10)



####################################################################
# set a few constants
# faster to read this in once, hard code stats
####################################################################
#data = pd.read_csv('train.csv')
#print(data.describe())

max_time = 16.1			# max time recorded
min_time = 0.                   # min time recorded

seg_length = 150000             # length of test segments

max_acoustic = 5515.            # max abs acoustic recorded
mean_acoustic = 4.51946757370
std_acoustic = 10.73570724951
std_2 = 2 * std_acoustic

cap = 551.5

fs = 4e6


############################################################################################
# features to try
# iterate through calculating features and checking correlation with target 
# several times - dumpy features that aren't useful
# this is what's left....
############################################################################################
features = [
	'power1', 
	'power2',
	'power3', 
	
	'time_to_failure', 'seg_id']



# set up dataframes to store features for training and testing data
train_features = features.copy()
train_features.remove('seg_id')
train_data = pd.DataFrame(columns=train_features)


test_features = features.copy()
test_features.remove('time_to_failure')
test_data = pd.DataFrame(columns=test_features)




##############################################################################################
# feature calculations
##############################################################################################
def get_fft(z):

	# calculate the fft on the segment
	frequencies = scipy.fftpack.fft(z)
	
	# get power
	n_nyquist = len(z) // 2
	power = np.abs(frequencies[1:n_nyquist]).real 

	'''
	power2 = np.sum(power[1500:2000]) 
	power3 = np.sum(power[2000:2500]) 
	
	power7 = np.sum(power[11000:11500]) 
	power8 = np.sum(power[11500:12000]) 
	
	power11 = np.sum(power[13000:13500]) 
	power12 = np.sum(power[13500:14000]) 
	power13 = np.sum(power[14000:14500]) 
	power14 = np.sum(power[14500:15000]) 
	'''

	power1 = np.sum(power[1500:2500])
	power2 = np.sum(power[11000:12000])
	power3 = np.sum(power[13000:15000])
	

	

	power_array = [power1, power2, power3]
	
	return power_array



def get_seg_stats(z):

	n_z = len(z)
	
	median = np.median(z)
	median_dev_abs = np.sum(np.abs(z - median)) 
	
	over_2std = (np.count_nonzero(np.abs(z) > 2 * std_acoustic))/ n_z
	
	stats_array = [median_dev_abs, over_2std]

	return stats_array
	




##########################################################################################################
# split the training file into segements matching 150k test files, calculate stats and save dataframe
##########################################################################################################

train = pd.read_csv('train.csv')
n_train = len(train)


step = 100000

for i in range(n_train):
#for i in range(5):

	start_position = step * i
	end_position = start_position + seg_length
	if end_position > n_train: break

	
	section = train[start_position:end_position]['acoustic_data'] - mean_acoustic
	section = np.clip(section, -cap, cap)
	time_to_failure = train.iloc[end_position]['time_to_failure']
	

	fft_array = get_fft(section)


	
	train_data.loc[str(i)] = ({
	'power1': fft_array[0], 
	'power2': fft_array[1],
	'power3': fft_array[2],
	
        'time_to_failure': time_to_failure })
    


# write to disk
train_data.to_csv('train_stats.csv')

print(train_data.describe())
max_values = train_data.max(axis=0)





##########################################################################################################
# read in files from test directory, calculate features and store in a dataframe
#############################################################################################################

print('prepping test data')
test_files = os.listdir('test')
n_test = len(test_files)



for i in range(n_test):
#for i in range(5):

	fname = test_files[i]
	id_ = fname.split('.')[0]

	fname = 'test/' + fname
	data = pd.read_csv(fname)
	seg = data['acoustic_data'] - mean_acoustic
	seg = np.clip(seg, -cap, cap)
	
	#print(id_)

	fft_array = get_fft(seg)

	
 
	test_data.loc[str(i)] = ({
	'power1': fft_array[0], 
	'power2': fft_array[1],
	'power3': fft_array[2],
	
        'seg_id': id_ })
    



# write to disk
test_data.to_csv('test_stats.csv')

print(test_data.describe())
print(test_data.dtypes)




#############################################################################################3
# plot features against target
##################################################################################################


data = pd.read_csv('train_stats.csv', index_col=0)
max_values = data.max(axis=0)

print(data.describe())


plot_features = ['power1', 'power2', 'power3']





for i in range(len(plot_features)):

	f = plot_features[i]
	fname = f + '.png'
	title = str(f)
	print(title)
	fig, ax = plt.subplots(figsize=(16, 16))
	plt.title = title

	z = data[f] / max_values[i] * 20 
	plt.plot(z, c='r')

	plt.plot(data['time_to_failure'])

	plt.savefig(fname)
	plt.show()



print(np.abs(data.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])








