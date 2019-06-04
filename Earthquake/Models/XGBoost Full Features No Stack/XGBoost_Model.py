
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 09:21:29 2019

# http://github.com/timestocome
# Linda MacPhee-Cobb


"""
################################################################################
# libraries
################################################################################
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt


##################################################################################
# print settings
##################################################################################
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 1000)

np.set_printoptions(precision=10)



###################################################################################
# read training data that was created with prep_data.py, 
# scale data as needed
# convert format
# split into train and hold out sets
###################################################################################
print('Reading data.............................')


# features selected in cross_validate.py
#features = ['gt_2std', 'gt_3std', 'power1', 'power2', 'power3', 'power4', 'median_dev_abs']


features = ['power1', 'power2', 'power3', 'power4', 'power5', 'power6', 'power7', 'power8', 'power9', 'power10', 'avg', 'std', 'median', 'var', 'kurt', 'skew', 'median_dev_abs', 'std_dev_abs', 'over_1std', 'over_2std', 'over_3std', 'energy_distance', 'work_distance', 'p_avg', 'p_std', 'p_median', 'p_var', 'p_kurt', 'p_hmean', 'p_gmean', 'p_skew', 'p_median_dev_abs', 'p_std_dev_abs']




# read in training data created in test_features
data = pd.read_csv('train_stats.csv', index_col=0)
print(list(data.columns.values))
#print(data.describe())

# used for plotting and error estimates
all_data = data[features]
all_label = data['time_to_failure']
dall_data = xgb.DMatrix(all_data, label=all_label)

n_train = 3800

train = data[0:n_train]
train = train.sample(frac=1.)

train_data = train[features]
train_label = train['time_to_failure']

print('train', train_data.shape, train_label.shape)


test = data[features][n_train:len(data)]
test_label = data['time_to_failure'][n_train:len(data)]

dtrain = xgb.DMatrix(train_data, label=train_label)
dtest = xgb.DMatrix(test, label=test_label)


print('test', test.shape, test_label.shape)

# read in submission data
submission_data = pd.read_csv('test_stats.csv', index_col='seg_id')
dtrain_submission_data = xgb.DMatrix(submission_data[features])

print('submission', submission_data.shape)





#########################################################################################
# set up and train model
#########################################################################################
print('training model')

param = { 'max_depth': 8, 'eta':0.2,  'colsample_bytree': 0.3, 'lambda':0.5, 'eval_metric': 'mae'}
evallist = [(dtest, 'eval'), (dtrain, 'train')]
n_rounds = 10



bst = xgb.train(param, dtrain, n_rounds, evallist)
bst.save_model('xgb.model')

print('model trained... saving..')


print('****************************************************************')
feature_values = bst.get_score(importance_type='gain')

import operator

sorted_fv = sorted(feature_values.items(), key=operator.itemgetter(1))

for k, v in sorted_fv:
	print('%s: %f' %(k, v))
print('****************************************************************')


bst.dump_model('dump_raw.txt')

#bst = xgb.Booster({'nthread': 4})  # init model
#bst.load_model('model.bin')  # load data








###############################################################################################
# plot predictions vs actual on training data and holdout data
###############################################################################################
predict_train = bst.predict(dall_data)
predict_y = all_label
predict_diff = (predict_train - predict_y )
predict_mae = np.sum(np.abs(predict_diff)) / len(data)

predict_test = bst.predict(dtest)
predict_test_diff = (predict_test - test_label )
predict_test_mae = np.sum(np.abs(predict_test_diff)) / len(test_label)
print('Total error:', predict_mae)
print('Test error:', predict_test_mae)


fig, ax = plt.subplots(figsize=(16, 16))
plt.plot(predict_train, c='r', label='Predicted')
plt.plot(predict_y, c='b', label='Actual', alpha=0.3)
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()




#############################################################################################
# retrain on full dataset and plot
#############################################################################################
print('----------------   all data  ---------------------------------------')



# used for plotting and error estimates
all_data = pd.read_csv('stats_data.csv', index_col=0)
all_data.sample(frac=1.)
all_data = data[features]
all_label = data['time_to_failure']
dall_data = xgb.DMatrix(all_data, label=all_label)


bst = xgb.train(param, dall_data, n_rounds, evallist)
bst.save_model('xgb.model')

print('model trained... saving..')


print('****************************************************************')
feature_values = bst.get_score(importance_type='gain')

import operator
sorted_fv = sorted(feature_values.items(), key=operator.itemgetter(1))

for k, v in sorted_fv:
	print('%s: %f' %(k, v))
print('****************************************************************')
bst.dump_model('dump_raw.txt')

#bst = xgb.Booster({'nthread': 4})  # init model
#bst.load_model('model.bin')  # load data

all_data = pd.read_csv('stats_data.csv', index_col=0)
all_data = data[features]
all_label = data['time_to_failure']
dall_data = xgb.DMatrix(all_data, label=all_label)

predict_train = bst.predict(dall_data)
predict_y = all_label
predict_diff = (predict_train - predict_y)
predict_mae = np.sum(np.abs(predict_diff)) / len(data)

print('Total error:', predict_mae)

# save train predictions for later use
np.savetxt('xgboost_train_predictions.csv', predict_train)


fig, ax = plt.subplots(figsize=(16, 16))
plt.plot(predict_train, c='r', label='Predicted')
plt.plot(predict_y, c='b', label='Actual', alpha=0.3)
plt.legend()
plt.grid(True)
plt.savefig('predictions.png')
plt.show()



#############################################################################################
# run submission data through best model and create submission file for kaggle
##############################################################################################

submission_data['time_to_failure'] = bst.predict(dtrain_submission_data)

submission_data = submission_data['time_to_failure']
print(submission_data.head())

submission_data.to_csv('xgboost_submission.csv', header='seg_id,time_to_failure')

print('-----------------  finished --------------------')



