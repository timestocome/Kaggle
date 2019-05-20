### Notes on https://www.kaggle.com/c/LANL-Earthquake-Prediction 

### Kaggle Contest to predict time from start to earthquake on lab created earthquakes

Data is in 150k long segments containing acoustic data from accelerometer and time to failure

Each segment measures 0.0375 seconds so any solution will have to create features and make predictions quickly to be useful


#### Things that didn't work:
LSTM, RNNs - it's not really that kind of a problem

Spectrograms - not enough data for a CNN

Breaking segments into smaller parts - overfitting

Breaking segments into separate quakes (16 ) before splitting into segemnts - idk? 

Augmenting data by: overlapping segments, adding noise - model answers were terrible


#### Things that did work:
Per many published papers standard practice is to calcuate several statistical features on each segment and run them into a model


#### Problems:
Statistical features are strongly correlated with each other causing overfitting if too many features are used

The dataset number of samples is small
