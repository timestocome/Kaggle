
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# prep data for model only calculating useful features

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.fftpack
import os




########################################################################################
# printing options
##########################################################################################
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)




###########################################################################################
# constants
# read in full data set and save a few things so they don't have to be recalced every time
###########################################################################################
# grab max, min, std, mean from describe
train = pd.read_csv('train.csv', dtype={'acoustic_data':np.int16, 'time_to_failure':np.float64})
print(train.shape)
print(train.describe())



Hz = 4e+6                         # recording speed
n_rows = 150000                   # length of test segments
data_scale = 5515.                # largest abs acoustic 
target_scale = 16.1               # largest time in data


# unscaled
data_std = 10.73571
data_mean = 4.519468
data_3_std = 3. * data_std
data_2_std = 2. * data_std



'''
# scaled
data_std = 1.946638e-3               # np.std(train['acoustic_data']/data_scale)
data_mean = 8.194864e-4		     # np.mean(train['acoustic_data']/data_scale)
data_3_std = 3. * data_std           # data_std * 3      
data_2_std = 2. * data_std           # data_std * 2
'''



##########################################################################################
# feature functions
##########################################################################################

def get_percent_over_std(s):

	n_s = len(s)
	over_1std = np.count_nonzero(np.abs(s) > data_std) / n_s
	over_2std = np.count_nonzero(np.abs(s) > data_2_std) / n_s
	over_3std = np.count_nonzero(np.abs(s) > data_3_std) / n_s

	return over_1std, over_2std, over_3std


def get_power(s):

	frequencies = scipy.fftpack.fft(section).real
	power = np.abs(frequencies) 
	
	# plotted power for all and eyeballed dips between peaks to get splits
	power1 = np.sum(power[1:2000])
	power2 = np.sum(power[2001:4000])
	power3 = np.sum(power[4001:8000])
	power4 = np.sum(power[8001:12000])

	return power1, power2, power3, power4




def get_deviation(s):
	# tried a slew of measures, this one performed the best
	median = np.median(s) 
	median_dev_abs = np.sum(np.abs(s - median)) 
	
	return median_dev_abs


############################################################################################
# read in training file and break into sections the same length as test data
# then compute what ever features appear to be useful
############################################################################################

train = pd.read_csv('train.csv', dtype={'acoustic_data':np.int16, 'time_to_failure':np.float64})
#print(train.shape)


n_samples = len(train)          # number of data points
n_step = 75000    		# amount of overlap used to expand training file


# things not yet tried
#weights = np.arange(0, n_rows)
# percent beyond one std,  max slope,  
# percent close to median, 



##########################################################################################
# features that appear useful, some are not used in models
##########################################################################################

features = [
             'gt_2std',
             'gt_3std',
             'power1',
             'power2',
             'power3',
             'power4',
             'median_dev_abs',
             'time_to_failure', 'seg_id']



# set up train, test data frames to store features
train_features = features.copy()
train_features.remove('seg_id')
data = pd.DataFrame(columns=train_features)

test_features = features.copy()
test_features.remove('time_to_failure')
test_data = pd.DataFrame(columns=test_features)


############################################################################################
# process training data
#############################################################################################



step = 0
n_loops = n_samples // n_step
for i in range(n_loops):
#for i in range(5):

	start_position = step
	end_position = step + n_rows

	if end_position >= n_samples: break
    
	section = train[start_position:end_position]['acoustic_data'] 


	gt_1std, gt_2std, gt_3std = get_percent_over_std(section)
	power1, power2, power3, power4 = get_power(section)
	median_dev_abs = get_deviation(section)
   	
   
	time_to_failure = train.iloc[end_position]['time_to_failure'] 


        
	data.loc[str(i)] = ({
		'gt_2std': gt_2std,
		'gt_3std': gt_3std,
	        'power1': power1,
	        'power2' : power2,
	        'power3' : power3,
	        'power4' : power4,
	        'median_dev_abs': median_dev_abs,
	        'time_to_failure': time_to_failure})
    
	step += n_step
	#print(i)





##################################################################################
# check train file as expected and save
#####################################################################################
print(data.head())
print(data.describe())

print('saving training data')
data.to_csv('stats_train.csv')





#############################################################################################
# process test data
############################################################################################


print('prepping test data')
test_files = os.listdir('test')
n_test = len(test_files)


#for i in range(5):
for i in range(n_test):

	fname = test_files[i]
	id_ = fname.split('.')[0]
	fname = 'test/' + fname
	section = pd.read_csv(fname)['acoustic_data'] 
	#section = np.loadtxt(fname)
	#print(section)
	#print(id_)
  

	gt_1std, gt_2std, gt_3std = get_percent_over_std(section)
	power1, power2, power3, power4 = get_power(section)
	median_dev_abs = get_deviation(section)


        
	test_data.loc[str(i)] = ({
		'gt_2std': gt_2std,
		'gt_3std': gt_3std,
	        'power1': power1,
	        'power2' : power2,
	        'power3' : power3,
	        'power4' : power4,
	        'median_dev_abs': median_dev_abs,
	        'seg_id': id_})
    


###################################################################################
# write test file to disk and check looks okay
#####################################################################################

test_data.to_csv('stats_test.csv')


# check test file okay
check = pd.read_csv('stats_test.csv', index_col=0)
print(check.head())
print(check.describe())
print(check.dtypes)



