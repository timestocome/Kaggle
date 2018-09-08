

# http://github.com/timestocome
# Kaggle Cats and Dogs data set


###################################################################################################
# data
###################################################################################################
import os, shutil
from os import listdir
import numpy as np
import re
from keras.preprocessing.image import ImageDataGenerator


# load up training and submission files and randomly sort them into 
# train, test, validate, submission and prep for network



# find files
path = os.getcwd()
train_data_path = path + '\\train'
submission_data_path = path + '\\submission'


# read in training data and split into cats and dogs
images = os.listdir(train_data_path)
cat_dir = train_data_path + '\\cat\\'
dog_dir = train_data_path + '\\dog\\'



if not os.path.exists(cat_dir):
    os.makedirs(cat_dir)
if not os.path.exists(dog_dir):
    os.makedirs(dog_dir)



for i in images:
    src = train_data_path + '\\' + i
    dst = ' '
    if re.match('cat\.(\d+)\.jpg', i):
        shutil.move(src, cat_dir)
    elif re.match('dog\.(\d+)\.jpg', i):
        shutil.move(src, dog_dir)
   

# split cat and dog files into train, test, validation
cats = os.listdir(cat_dir)
dogs = os.listdir(dog_dir)

# shuffle images
np.random.shuffle(cats)
np.random.shuffle(dogs)

# how many of each
n_cats = len(cats)
n_dogs = len(dogs)

# how many for each set
n_train = n_cats
n_test = n_cats // 10
n_validate = n_cats // 10


# create test, validate dirs
cat_test = path + '\\test\\cat'
cat_validate = path + '\\validate\\cat'

if not os.path.exists(cat_test):
    os.makedirs(cat_test)
if not os.path.exists(cat_validate):
    os.makedirs(cat_validate)

dog_test = path + '\\test\\dog'
dog_validate = path + '\\validate\\dog'

if not os.path.exists(dog_test):
    os.makedirs(dog_test)
if not os.path.exists(dog_validate):
    os.makedirs(dog_validate)


# move files if not already moved into various directories
for i in range(n_test):
    
    src = cat_dir + cats[i]
    dst = cat_test
    shutil.move(src, dst)
    
    src = dog_dir + dogs[i]
    dst = dog_test
    shutil.move(src, dst)
    
    src = cat_dir + cats[n_test + i]
    dst = cat_validate
    shutil.move(src, dst)
    
    src = dog_dir + dogs[n_test + i]
    dst = dog_validate
    shutil.move(src, dst)


n_cat_test = len(os.listdir(cat_test))
n_cat_validate = len(os.listdir(cat_validate))
n_cat_train = len(os.listdir(cat_dir))

n_dog_test = len(os.listdir(dog_test))
n_dog_validate = len(os.listdir(dog_validate))
n_dog_train = len(os.listdir(dog_dir))


# sanity check data counts
print('cats and dogs', n_cats, n_dogs)
print(n_cat_validate, n_dog_validate)
print(n_cat_test, n_dog_test)
print(n_cat_train, n_dog_train)





