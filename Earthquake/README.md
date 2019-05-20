Notes on https://www.kaggle.com/c/LANL-Earthquake-Prediction 

Kaggle Contest to predict time from start to earthquake on lab created earthquakes

Data is in 150k long segments containing acoustic data from accelerometer and time to failure

Each segment measures 0.0375 seconds so any solution will have to create features and make predictions quickly to be useful


Things that didn't work:
LSTM, RNNs
Breaking segments into smaller parts
Augmenting data by: overlapping segments, adding noise
