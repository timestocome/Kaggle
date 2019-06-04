

# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# read in data, split training data, create statistics for each 
# section of data ( train and test ) and save stastical info to 
# dist to use in models


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
#####################################################################
#data = pd.read_csv('train.csv')
#print(data.describe())

max_time = 16.2			# max time recorded
min_time = 0.                   # min time recorded
seg_length = 150000.            # length of test segments
max_acoustic = 5500.            # max acoustic recorded



###################################################################
train_files = os.listdir('train')
test_files = os.listdir('test')

n_train = len(train_files)
n_test = len(test_files)

print('train', len(train_files))
print('test', len(test_files))





features = ['max', 
	'min', 
	'i_max', 
	'i_min', 
	'cum_abs_sum', 
	'min_dx', 
	'i_min_dx', 
	'max_dx', 
	'i_max_dx',
	'max_abs_dx',
	'i_max_abs_dx', 
	'var', 
	'std', 
	'skew', 
	'dc_freq', 
	'dc_power',
	'pow1', 
	'pow2', 
	'pow3', 
	'pow4', 
	'max_power', 
	'max_power_loc',
	'sum_power',
	'kurtosis', 
	'avg', 
	'median', 
	'median_dev_abs', 
	'std_dev_abs', 
        'time_to_failure', 'seg_id']


train_features = features.copy()
train_features.remove('seg_id')
train_data = pd.DataFrame(columns=train_features)


test_features = features.copy()
test_features.remove('time_to_failure')
test_data = pd.DataFrame(columns=test_features)






def get_max(s):
	max_ = np.max(s)
	i_max = np.argmax(s) / seg_length
	return max_, i_max


def get_min(s):
	min_ = np.min(s)
	i_min = np.argmin(s) / seg_length
	return min_, i_min



def get_dx(s):

	dx = s - s.shift(1)
	dx.dropna(axis=0, inplace=True)
	min_dx = np.min(dx) 
	min_dx_loc = np.argmin(dx) / seg_length
	max_dx = np.max(dx) 
	max_dx_loc = np.argmax(dx) / seg_length
	max_abs_dx = np.max(np.abs(dx))
	max_abs_dx_loc = np.argmax(np.abs(dx)) / seg_length
	
	return min_dx, max_dx, max_abs_dx, max_dx_loc, min_dx_loc, max_abs_dx_loc


def get_stats(s):

	var_ = np.var(s)
	std_ = np.std(s)
	skew_ = stats.skew(s)  / 5.

	return var_, std_, skew_



def get_fft(s):

	cut_off = 12500
	frequencies = scipy.fftpack.fft(s)
	power = np.abs(frequencies[1:cut_off]).real 
	dc_freq = frequencies[0].real / 150.
	dc_power = power[0] / 8.
	power1 = np.sum(power[1:2000]) / (cut_off * 3.)
	power2 = np.sum(power[2001:4000]) / (cut_off * 5.)
	power3 = np.sum(power[4001:8000])  / (cut_off * 9.)
	power4 = np.sum(power[8001:12000]) / (cut_off * 8.)
	max_power = np.max(power) / 110.
	max_power_loc = np.argmax(power) / cut_off
	sum_power = np.sum(power) / (cut_off * 20.)

	return dc_freq, dc_power, power1, power2, power3, power4, max_power, max_power_loc, sum_power


def get_kurt(s):
	
	kurt = stats.kurtosis(s) / 900.

	return kurt


def get_sum(s):

	sum_ = np.sum(s)
	cum_abs_sum = np.sum(np.abs(s)) / 894.
	
	return sum_, cum_abs_sum


def get_dev(s):

	avg_ = np.average(s) * 1000.
	std_ = np.std(s)
	median = np.median(s)
	median_dev_abs = np.sum(np.abs(s - median)) / seg_length
	std_dev_abs = np.sum(np.abs(s - std_)) / seg_length

	return avg_, median, std_dev_abs, median_dev_abs 




for i in range(n_train):
#for i in range(5):

    fname = train_files[i]
    fname = 'train/' + fname
    data = pd.read_csv(fname)
    seg = data['acoustic_data'] / max_acoustic
    target = data.iloc[-1]['time_to_failure'] / max_time
    dc_f, dc_p, p1, p2, p3, p4, max_power, max_power_loc, sum_power = get_fft(seg)
    max_, i_max = get_max(seg)
    min_, i_min = get_min(seg)
    min_dx, max_dx, max_abs_dx, max_dx_loc, min_dx_loc, max_abs_dx_loc = get_dx(seg)
    var, std, skew = get_stats(seg)
    kurt = get_kurt(seg)
    avg_, median, std_dev_abs, median_dev_abs = get_dev(seg)
    sum_, cum_abs_sum = get_sum(seg)

   

    train_data.loc[str(i)] = ({
	'max': max_,
	'min': min_,
	'i_max': i_max,
	'i_min': i_min,
	'cum_abs_sum': cum_abs_sum,
	'min_dx' : min_dx,
	'i_min_dx': min_dx_loc,
	'max_dx': max_dx,
	'i_max_dx': max_dx_loc,
	'max_abs_dx': max_abs_dx,
	'i_max_abs_dx': max_abs_dx_loc,
	'var': var,
	'std': std,
	'skew': skew,
	'dc_freq': dc_f,
	'dc_power': dc_p,
	'pow1' : p1,
	'pow2' : p2,
	'pow3': p3, 
	'pow4': p4,
	'max_power': max_power,
	'max_power_loc' : max_power_loc,
	'sum_power': sum_power,
	'kurtosis': kurt,
	'avg': avg_,
	'median': median,
	'median_dev_abs': median_dev_abs,
	'std_dev_abs' : std_dev_abs,
        'time_to_failure': target })
    

train_data.to_csv('train_stats.csv')


print(train_data.describe())
print(train_data.dtypes)



# get correlation with target
print(np.abs(train_data.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])




'''

for i in range(n_test):
#for i in range(5):

    fname = test_files[i]
    id_ = fname.split('.')[0]
    fname = 'test/' + fname
    data = pd.read_csv(fname)
    seg = data['acoustic_data']

    #print(seg)
    #print(id_)


    dc_f, p1, p2, p3, p4, max_power, sum_power = get_fft(seg)
    max_, i_max = get_max(seg)
    min_, i_min = get_min(seg)
    cum_abs_sum = get_cum_abs_sum(seg)
    min_dx, max_dx, max_dx_loc, min_dx_loc = get_dx(seg)
    var, std, skew = get_stats(seg)
    kurt = get_kurt(seg)
    avg = get_avg(seg)
    median, std_dev_abs, median_dev_abs = get_dev(seg)


 
    test_data.loc[str(i)] = ({
	'max' : max_,
        'min' : min_,
	'i_max' : i_max,
	'i_min' : i_min,
	'cum_abs_sum' : cum_abs_sum,
        'min_dx' : min_dx,
        'i_min_dx' : min_dx_loc,
        'max_dx' : max_dx,
        'i_max_dx' : max_dx_loc,
        'var' : var,
        'std' : std,
        'skew' : skew,
	'dc_freq' : dc_f,
	'pow1': p1,
	'pow2': p2, 
	'pow3': p3,
	'pow4': p4,
	'max_power': max_power,
	'sum_power': sum_power,
	'kurtosis': kurt,
	'avg': avg,
	'median' : median,
	'median_dev_abs': median_dev_abs,
	'std_dev_abs': std_dev_abs,
        'seg_id': id_ })
    

test_data.to_csv('test_stats.csv')

print(test_data.describe())
print(test_data.dtypes)


'''















