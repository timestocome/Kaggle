
# http://github.com/timestocome
# Kaggle Cats and Dogs data set


###################################################################################################
# data
###################################################################################################
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


path = os.getcwd()

train_dir = path + '\\train'
test_dir = path + '\\test'
validate_dir = path + '\\validate'
submission_dir = path + '\\submission'



# load and cleanup images, add augmented images, resize to 150, 150

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
    class_mode='binary'
)

# used as hold out data to be sure we aren't fitting our
# model to the validation data
test_generator = submission_datagen.flow_from_directory(
    test_dir,
    target_size=(width, height),
    batch_size=batch_size,
    class_mode='binary'
)

# used to tell when training bottoms out and begins to overfit 
# (memorize data)
validate_generator = submission_datagen.flow_from_directory(
    validate_dir,
    target_size=(width, height),
    batch_size=batch_size,
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
    steps_per_epoch=100,
    epochs=300,
    validation_data = validate_generator,
    validation_steps=50
)



model.save('cats_and_dogs.h5')

# test against some hold out data
test_loss, test_accuracy = model.evaluate_generator(test_generator, steps=10 )
print('Hold out accuracy: %lf' %(test_accuracy))




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

###########################################################################################################
# plotting
###########################################################################################################

# plot progress
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, 'bo', label='Training')
plt.plot(epochs, val_acc, 'b', label='Validation')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()






