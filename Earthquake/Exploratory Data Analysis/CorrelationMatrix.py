

# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction


# take a look at features relationship to target and each other

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix



# read in train data feature file
train = pd.read_csv('train_stats.csv', index_col=0)

print(train.head())
print(list(train.columns.values))


features = ['power1', 'power2', 'power3', 'power4', 'power5', 'power6', 'power7', 'power8', 'power9', 'power10', 'avg', 'std', 'median', 'var', 'kurt', 'skew', 'median_dev_abs', 'std_dev_abs', 'over_1std', 'over_2std', 'over_3std', 'energy_distance', 'work_distance', 'p_avg', 'p_std', 'p_median', 'p_var', 'p_kurt', 'p_hmean', 'p_gmean', 'p_skew', 'p_median_dev_abs', 'p_std_dev_abs', 'time_to_failure']


# compare to target
print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])

'''
time_to_failure     1.000000
over_1std           0.589309
energy_distance     0.576763
over_2std           0.411490
median_dev_abs      0.341756
work_distance       0.340329
over_3std           0.326009
p_skew              0.291649
p_kurt              0.267540
p_hmean             0.261703
power5              0.254176
power6              0.252139
power4              0.242674
power8              0.235833
power7              0.233517
p_gmean             0.230341
p_median_dev_abs    0.228609
p_std               0.228301
p_avg               0.218694
std                 0.217069
p_median            0.211109
p_std_dev_abs       0.205365
power2              0.193536
power9              0.193179
power3              0.189012
std_dev_abs         0.178612
power10             0.145921
power1              0.136177
kurt                0.109619
var                 0.105175
p_var               0.104060
avg                 0.031315
skew                0.015574
median              0.004445
'''





# compare features to each other


features = ['avg', 'kurt', 'median_dev_abs', 'std_dev_abs', 'over_1std','p_hmean', 'p_skew', 'time_to_failure']


train = train[features]
print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['time_to_failure'])
#print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['median_dev_abs'])
#print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['p_skew'])
#print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['p_hmean'])
#print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['std_dev_abs'])
#print(np.abs(train.corr()).sort_values('time_to_failure', ascending=False)['kurt'])



# meh, idk about this part, looks good but selected features didn't perform well in model


# https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/
corr_matrix = train.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
drop_index = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]


print(drop_index)
drop_columns = ['power2', 'power3', 'power4', 'power5', 'power6', 'power7', 'power8', 'power9', 'power10', 'std', 'var', 'median_dev_abs', 'std_dev_abs', 'over_2std', 'over_3std', 'energy_distance', 'work_distance', 'p_avg', 'p_std', 'p_median', 'p_var', 'p_hmean', 'p_gmean', 'p_skew', 'p_median_dev_abs', 'p_std_dev_abs']


train = train.drop(columns=drop_columns)

print(train.head())





