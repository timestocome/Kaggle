### Notes on https://www.kaggle.com/c/LANL-Earthquake-Prediction 

### Kaggle Contest to predict time from start to earthquake on lab created earthquakes


Training data is a 9.6gb file that breaks down into 16 quakes, or 4096 segments if they aren't overlapped. Data contains acoustic signal from accelerometer and time to failure


Test data is 150k long segments containing acoustic data from accelerometer. These segments are randomly chosen from 
several quakes to prevent people lining them up and manually entering time to failure


Each segment measures 0.0375 seconds so any solution will have to create features and make predictions quickly to be useful

My most successful models used a small feature set with XGBoost. (power, deviation from mean, number of zero crossings)
Others found slope trend to be useful, added white noise, blended models

#### Things that didn't work for me:
LSTM, RNNs - it's not really that kind of a problem

1D Convolutional - Even with 64Gb RAM and a 1070 8GB GPU the computer couldn't handle it. I tried breaking the 
	data up into 30k segments but the computer still choked

Spectrograms - not enough data to train a CNN

Histograms - this was disappointing it was a big help with stock market predictions

Breaking segments into smaller parts, overlapping segments at a step of 75k caused overfitting. One contestant temporarily gained first place with random selections but that didn't work for me.

Augmenting data by: adding 1% noise to unscaled acoustic data - model answers were terrible

Breaking segments into separate quakes (16) before splitting into segemnts - models showed prediction of quake, then jumped the gun predicting the start of the next quake before this one settled down. The hope was that splitting the quakes would remove this artifact but it did not.

Using the difference between points instead of actual points, models had about the same accuracy so I skipped the extra step

Scaling data, since I used RandomForest ensembles and XGBoost data didn't require scaling, this saved a step and a lot of time. I did give it a try but it made no difference in model accuracy.

Entropy, it seemed like it would make a great predictor, growing or shrinking, as the quake neared but it didn't pan out.

Clusters - I tried creating 2 clusters which did show the beginning and end of a quake but in the model it didn't remove the artifact at the end-beginning of the quakes and also caused severe over fitting and it required scaling data ( less is more when the data samples coming in are only .0375 seconds in length

Taking the square root of features and other feature augmentations, like adding same features that were calculated on the end of the segment and full segment





#### Things that did work:
Per many published papers standard practice is to calcuate several statistical features on each segment and run them into a model.

Statistical measures of the increased bouncing and size of the acoustical signal were the best predictors: Absolute Deviation from Norm, Number of Zero Crossings were the most predictive



#### Problems:
Statistical features are strongly correlated with each other causing overfitting if too many features are used

Too many features, too few samples. The dataset number of samples (4096) is small compared to length of sample (150000) making it necessary to reduce samples by statistical or other methods. 

Often one feature would overwhelm the other features contributing 85%+ to the model. When this occurs the plots and error scores would look good but the model never did well against the test data.

MAE tends to push the predictions towards the mean more strongly than other error functions. In order to improve the score you gave up estimating peaks and valleys in the data. In addition there were a couple minor quakes that confused the predictors


#### Scores:
Winning score: 2.26589

My score: 2.56844 ( top 27% ) 

### Models:
Models: 
- Ensemble Random Forest Regressor ( Small feature set submission )
- XGBoost Model predictions then fed into a Linear Regression model (Stacked Model)
- (XGBoost Few Features)
- (XGBoost Full Features, no stack)






