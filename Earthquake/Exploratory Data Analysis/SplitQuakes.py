




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



train = pd.read_csv('train.csv')


train['time_dx'] = train['time_to_failure'] - train['time_to_failure'].shift(1)

'''
breaks = train.nlargest(16, 'time_dx')
print(breaks)
'''

cap = train['acoustic_data'].max() // 10
acoustic_mean = train['acoustic_data'].mean()

print(cap, acoustic_mean)
train['acoustic_data'] = train['acoustic_data'] - acoustic_mean
train['acoustic_data'] = np.clip(train['acoustic_data'], -cap, cap)



quake17 = train[621985673:-1]
quake16 = train[585568144:621985673]
quake15 = train[528777115:585568144]
quake14 = train[495800225:528777115]
quake13 = train[461811623:495800225]
quake12 = train[419368880:461811623]
quake11 = train[375377848:419368880]
quake10 = train[338276287:375377848]
quake9 = train[307838917:338276287]
quake8 = train[245829585:307838917]
quake7 = train[218652630:245829585]
quake6 = train[187641820:218652630]
quake5 = train[138772453:187641820]
quake4 = train[104677356:138772453]
quake3 = train[56566574:104677356]
quake2 = train[50085878:56566574]
quake1 = train[0:5656574] 



quakes = [quake1, quake2, quake3, quake4, quake5, quake6, quake7, quake8, 
	quake9, quake10, quake11, quake12, quake13, quake14, quake15, quake16, 
	quake17]




###########################################################################################
# plot data but only every 100th point to keep from overwheling computer
##########################################################################################
for i in range(len(quakes)):

	print(i)
	print(quakes[i].describe())

	fig, ax1 = plt.subplots(figsize=(32,16))
	plt.title('Full data 1%')

	plt.plot(quakes[i]['acoustic_data'][::1000], color='r')
	ax1.set_ylabel('acoustic data', color='r')
	ax1.set_ylim(-500, 500)

	ax2 = ax1.twinx()
	plt.plot(quakes[i]['time_to_failure'][::1000], color='b')
	ax2.set_ylabel('time to failure', color='b')

	plt.plot(np.cumsum(quakes[i]['acoustic_data'], color='g')

	plt.grid(True)

	fname = 'quakes_' + str(i) + '.png'
	plt.savefig(fname)
	plt.show()










