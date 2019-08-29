

# http://github.com/timestocome
# Benchmark from molecules contest


# score = 0.784

import numpy as np
import pandas as pd
import os



###############################################################################
# read in data files
###############################################################################


submission = pd.read_csv('Data/test.csv', index_col=0)

print('submission file length', len(submission))

jhc1 = pd.read_csv('submission_jhc1.csv', index_col=0)
jhn1 = pd.read_csv('submission_jhn1.csv', index_col=0)
jhc2 = pd.read_csv('submission_jhc2.csv', index_col=0)
jhh2 = pd.read_csv('submission_jhh2.csv', index_col=0)
jhn2 = pd.read_csv('submission_jhn2.csv', index_col=0)
jhc3 = pd.read_csv('submission_jhc3.csv', index_col=0)
jhh3 = pd.read_csv('submission_jhh3.csv', index_col=0)
jhn3 = pd.read_csv('submission_jhn3.csv', index_col=0)


###############################################################################
# patch files together and sort
###############################################################################

combined = pd.concat([jhc1, jhn1, jhc2, jhh2, jhn2, jhc3, jhh3, jhn3])

print('combined file length', len(combined))

sorted_combined = combined.sort_index()
print(sorted_combined.columns.values)
n_samples = len(combined)

###############################################################################
# sanity check against prior entry
###############################################################################

previous_submission = pd.read_csv('previous_best.csv')
coupling_types = ['1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']


diff = np.sum(np.abs(combined['scalar_coupling_constant'] - previous_submission['scalar_coupling_constant'])) / n_samples

print('total diff', diff)



sorted_combined['previous_guess'] = previous_submission['scalar_coupling_constant']

combined_jhc1 = sorted_combined[sorted_combined['type'] == '1JHC']
jhc1_diff = np.sum(np.abs(combined_jhc1['scalar_coupling_constant'] - combined_jhc1['previous_guess'])) / len(combined_jhc1)
print('1JHC', jhc1_diff, len(jhc1))


combined_jhc2 = sorted_combined[sorted_combined['type'] == '2JHC']
jhc2_diff = np.sum(np.abs(combined_jhc2['scalar_coupling_constant'] - combined_jhc2['previous_guess'])) / len(combined_jhc2)
print('2JHC', jhc2_diff, len(jhc2))


combined_jhc3 = sorted_combined[sorted_combined['type'] == '3JHC']
jhc3_diff = np.sum(np.abs(combined_jhc3['scalar_coupling_constant'] - combined_jhc3['previous_guess'])) / len(combined_jhc3)
print('3JHC', jhc3_diff, len(jhc3))


combined_jhh2 = sorted_combined[sorted_combined['type'] == '2JHH']
jhh2_diff = np.sum(np.abs(combined_jhh2['scalar_coupling_constant'] - combined_jhh2['previous_guess'])) / len(combined_jhh2)
print('2JHH', jhh2_diff, len(jhh2))


combined_jhh3 = sorted_combined[sorted_combined['type'] == '3JHH']
jhh3_diff = np.sum(np.abs(combined_jhh3['scalar_coupling_constant'] - combined_jhh3['previous_guess'])) / len(combined_jhh3)
print('2JHH', jhh3_diff, len(jhh3))



combined_jhn1 = sorted_combined[sorted_combined['type'] == '1JHN']
jhn1_diff = np.sum(np.abs(combined_jhn1['scalar_coupling_constant'] - combined_jhn1['previous_guess'])) / len(combined_jhn1)
print('1JHN', jhn1_diff, len(jhn1))



combined_jhn2 = sorted_combined[sorted_combined['type'] == '2JHN']
jhn2_diff = np.sum(np.abs(combined_jhn2['scalar_coupling_constant'] - combined_jhn2['previous_guess'])) / len(combined_jhn2)
print('2JHN', jhn2_diff, len(jhn2))




combined_jhn3 = sorted_combined[sorted_combined['type'] == '3JHN']
jhn3_diff = np.sum(np.abs(combined_jhn3['scalar_coupling_constant'] - combined_jhn3['previous_guess'])) / len(combined_jhn3)
print('3JHN', jhn3_diff, len(jhn3))

'''

############################################################
combined = sorted_combined[['id', 'scalar_coupling_constant']]
combined.to_csv('addedfeatures_submission.csv', index=False)


###########################################################
combined = sorted_combined[['id', 'sum_sc']]
combined.columns = ['id', 'scalar_coupling_constant']
combined.to_csv('sum_sc_submission.csv', index=False)


###########################################################
combined = sorted_combined[['id', 'fc']]
combined.columns = ['id', 'scalar_coupling_constant']
combined.to_csv('fc_submission.csv', index=False)

'''



















