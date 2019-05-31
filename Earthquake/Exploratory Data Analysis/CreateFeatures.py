
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# create a slew of statistical features based on input data
# input data is too large 150k to sample size 4096 to create a model
# input size needs to be cut way down

##############################################################
# libraries used
################################################################

import numpy as np
import scipy.stats as stats
import scipy.fftpack
import pandas as pd
import os


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
std_3 = 3 * std_acoustic




############################################################################################
# features to try
# iterate through calculating features and checking correlation with target 
# several times - dumpy features that aren't useful
# this is what's left....
############################################################################################
features = ['power1', 
	'power2', 
	'power3', 
	'power4', 
	'power5', 
	'power6', 
	'power7', 
	'power8', 
	'power9', 
	'power10',
	'avg', 
	'std', 
	'median', 
	'var', 
	'kurt', 
	'skew', 
	'median_dev_abs', 
	'std_dev_abs', 
	'over_1std', 
	'over_2std', 
	'over_3std', 
	'energy_distance', 
	'work_distance', 
	'p_avg', 
	'p_std', 
	'p_median', 
	'p_var', 
	'p_kurt', 
	'p_hmean', 
	'p_gmean', 
	'p_skew', 
	'p_median_dev_abs', 
	'p_std_dev_abs', 
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

	n_nyquist = len(z) // 2
	
	# get power
	power = np.abs(frequencies[1:n_nyquist]).real 

	# average frequency - doesn't appear to be useful in models
	dc_freq = frequencies[0].real 
	dc_power = power[0] 

	power1 = np.sum(power[1:2000]) 
	power2 = np.sum(power[2001:4000]) 
	power3 = np.sum(power[4001:6000]) 
	power4 = np.sum(power[6001:8000]) 
	power5 = np.sum(power[8001:10000])
	power6 = np.sum(power[10001:12000])
	power7 = np.sum(power[12001:14000])
	power8 = np.sum(power[14001:16000])
	power9 = np.sum(power[16001:18000])
	power10 = np.sum(power[18001:20000])
	
	power_array = [power1, power2, power3, power4, power5, power6, power7, power8, power9, power10]
	
	return power_array



def get_seg_stats(z):

	n_z = len(z)
	avg = np.average(z)
	std = np.std(z)
	median = np.median(z)
	var = np.var(z)
	kurt = stats.kurtosis(z)
	gmean = stats.gmean(z)
	skew = stats.skew(z)
	median_dev_abs = np.sum(np.abs(z - median)) 
	std_dev_abs = np.sum(np.abs(z - std)) 

	over_1std = np.count_nonzero(np.abs(z) > std_acoustic) / n_z
	over_2std = np.count_nonzero(np.abs(z) > 2 * std_acoustic) / n_z
	over_3std = np.count_nonzero(np.abs(z) > 3 * std_acoustic) / n_z
	

	base = [mean_acoustic] * len(z)
	energy_distance = stats.energy_distance(z, base)
	work_distance = stats.wasserstein_distance(z, base)
	
	stats_array = [avg, std, median, var, kurt, skew, median_dev_abs, std_dev_abs,
			over_1std, over_2std, over_3std, energy_distance, work_distance]

	return stats_array
	

	
def get_fft_stats(z):

	avg = np.average(z)
	std = np.std(z)
	median = np.median(z)
	var = np.var(z)
	kurt = stats.kurtosis(z)
	hmean = stats.hmean(z)
	gmean = stats.gmean(z)
	skew = stats.skew(z)
	median_dev_abs = np.sum(np.abs(z - median)) 
	std_dev_abs = np.sum(np.abs(z - std)) 

	
	stats_array = [avg, std, median, var, kurt, hmean, gmean, skew, median_dev_abs, std_dev_abs]

	return stats_array









##########################################################################################################
# split the training file into segements matching 150k test files, calculate stats and save dataframe
##########################################################################################################

train = pd.read_csv('train.csv')
n_train = len(train)

step = 150000

for i in range(n_train):
#for i in range(5):

	start_position = step * i
	end_position = start_position + seg_length
	if end_position > n_train: break


	section = train[start_position:end_position]['acoustic_data']
	time_to_failure = train.iloc[end_position]['time_to_failure']
	

	fft_array = get_fft(section)
	seg_stats = get_seg_stats(section)
	fft_stats = get_fft_stats(fft_array)



	train_data.loc[str(i)] = ({
	'power1': fft_array[0], 
	'power2': fft_array[1], 
	'power3': fft_array[2], 
	'power4': fft_array[3], 
	'power5': fft_array[4], 
	'power6': fft_array[5], 
	'power7': fft_array[6], 
	'power8': fft_array[7], 
	'power9': fft_array[8], 
	'power10': fft_array[9],
	'avg': seg_stats[0], 
	'std': seg_stats[1], 
	'median': seg_stats[2], 
	'var': seg_stats[3], 
	'kurt': seg_stats[4], 
	'skew': seg_stats[5], 
	'median_dev_abs': seg_stats[6], 
	'std_dev_abs': seg_stats[7], 
	'over_1std': seg_stats[8], 
	'over_2std': seg_stats[9], 
	'over_3std': seg_stats[10], 
	'energy_distance': seg_stats[11], 
	'work_distance': seg_stats[12], 
	'p_avg': fft_stats[0], 
	'p_std': fft_stats[1], 
	'p_median': fft_stats[2], 
	'p_var': fft_stats[3], 
	'p_kurt': fft_stats[4], 
	'p_hmean': fft_stats[5], 
	'p_gmean': fft_stats[6], 
	'p_skew': fft_stats[7], 
	'p_median_dev_abs': fft_stats[8], 
	'p_std_dev_abs': fft_stats[9], 
        'time_to_failure': time_to_failure })
    


# write to disk
train_data.to_csv('train_stats.csv')

print(train_data.describe())
print(train_data.dtypes)


# check correlations - dump obviously bad features -
# don't rely on this alone it doesn't catch interactions between features
# get correlation with target
print(np.abs(train_data.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])




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
	seg = data['acoustic_data']

	#print(id_)



	fft_array = get_fft(seg)
	seg_stats = get_seg_stats(seg)
	fft_stats = get_fft_stats(fft_array)

 
	test_data.loc[str(i)] = ({
	'power1': fft_array[0], 
	'power2': fft_array[1], 
	'power3': fft_array[2], 
	'power4': fft_array[3], 
	'power5': fft_array[4], 
	'power6': fft_array[5], 
	'power7': fft_array[6], 
	'power8': fft_array[7], 
	'power9': fft_array[8], 
	'power10': fft_array[9],
	'avg': seg_stats[0], 
	'std': seg_stats[1], 
	'median': seg_stats[2], 
	'var': seg_stats[3], 
	'kurt': seg_stats[4], 
	'skew': seg_stats[5], 
	'median_dev_abs': seg_stats[6], 
	'std_dev_abs': seg_stats[7], 
	'over_1std': seg_stats[8], 
	'over_2std': seg_stats[9], 
	'over_3std': seg_stats[10], 
	'energy_distance': seg_stats[11], 
	'work_distance': seg_stats[12], 
	'p_avg': fft_stats[0], 
	'p_std': fft_stats[1], 
	'p_median': fft_stats[2], 
	'p_var': fft_stats[3], 
	'p_kurt': fft_stats[4], 
	'p_hmean': fft_stats[5], 
	'p_gmean': fft_stats[6], 
	'p_skew': fft_stats[7], 
	'p_median_dev_abs': fft_stats[8], 
	'p_std_dev_abs': fft_stats[9], 
        'seg_id': id_ })
    



# write to disk
test_data.to_csv('test_stats.csv')

print(test_data.describe())
print(test_data.dtypes)
















