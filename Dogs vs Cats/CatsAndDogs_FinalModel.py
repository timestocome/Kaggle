
# http://github.com/timestocome
# Kaggle Cats and Dogs data set


# take the best model setup during training and re-train on full training data set
# change locations of submission files to be 'cat' or 'dog' depending on prediction




###################################################################################################
# data
###################################################################################################
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import shutil
import re


path = os.getcwd()

train_dir = path + '\\train_all'
submission_dir = path + '\\submission'
output_cats_dir = path + '\\output_cats'
output_dogs_dir = path + '\\output_dogs'



# sort training data into cats and dogs
# read in training data and split into cats and dogs
images = os.listdir(train_dir)
cat_dir = train_dir + '\\cat\\'
dog_dir = train_dir + '\\dog\\'



if not os.path.exists(cat_dir):
    os.makedirs(cat_dir)
if not os.path.exists(dog_dir):
    os.makedirs(dog_dir)


'''
for i in images:
    src = train_dir + '\\' + i
    dst = ' '
    if re.match('cat\.(\d+)\.jpg', i):
        shutil.move(src, cat_dir)
    elif re.match('dog\.(\d+)\.jpg', i):
        shutil.move(src, dog_dir)
   
'''


# check files moved
n_cat_train = len(os.listdir(cat_dir))
n_dog_train = len(os.listdir(dog_dir))

print('cats %d, dogs %d' % (n_cat_train, n_dog_train))




# load and cleanup images, add augmented images, resize to 250, 250

width = height = 250
batch_size = 20


# scale image data 0-1
submission_datagen = ImageDataGenerator(
    rescale=1./255,
    )

# scale and create augmented images for training
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary'
)


# Note:
# Even though the final images are not labeled we need to put 
# a label on them or the generator will not import them
# so label them all 'cats' ( or dogs ) and use the final
# output to state if they are corrrect cats or < .50% likely
# to be dogs
submission_generator = submission_datagen.flow_from_directory(
    submission_dir,
    target_size=(width, height),
    batch_size=1,
    shuffle=False,
    class_mode=None
)







#####################################################################################################
# model
# Deep Learning with Python ( Manning ), Chapter 5.5
####################################################################################################
from keras import layers
from keras import models
from keras import optimizers


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(width, height, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])




#######################################################################################################
# training
#######################################################################################################

history = model.fit_generator(
    train_generator,
    steps_per_epoch=130,
    epochs=300,
)




model.save('cats_and_dogs_final.h5')



# get data for submission)
n_submissions = len(submission_generator.filenames)
print('n_submissions', n_submissions)
predictions = model.predict_generator(submission_generator, n_submissions)
np.savetxt('predictions.csv', predictions)
print('predictions', predictions)
print('n_predictions', len(predictions))




# dog == 1, cat == 0
submission_prediction = np.where(predictions < .5, 1, 0)
submission_data = zip(submission_generator.filenames, submission_prediction)
#for i in submission_data: print(i)

f = open('submission_data.txt', 'w')
for i in submission_data:
    f.write(str(i))
f.close()



# find out if predicted cat or dog and move file to proper dir
from shutil import copyfile


submission_files = os.listdir(submission_dir + '\\' + 'cat')
files = submission_generator.filenames


for i in range(n_submissions):

    src = submission_dir + '\\' + files[i]
    dst = path + '\\output_cats\\' 

    if predictions[i] < .5:
        dst = path + '\\output_dogs\\'

    shutil.move(src, dst)



print('.... completed')




