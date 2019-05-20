
# https://github.com/timestocome
# https://www.kaggle.com/c/LANL-Earthquake-Prediction

import numpy as np
import pandas as pd

train = pd.read_csv('train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

print(train.shape)


n_rows = 150000
n_segments = train.shape[0] // n_rows

print(n_segments)


for i in range(n_segments):

    seg = train[i*n_rows: i*n_rows + n_rows]
    fname = 'segment_' + str(i) + '.csv'
    print(fname)
    seg.to_csv(fname)



print('finished')
