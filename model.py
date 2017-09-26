from keras.layers import Flatten, Convolution2D, Dense, Lambda, Cropping2D, Dropout
from keras.models import Sequential, load_model
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split
import csv
import cv2
import numpy as np


samples = []
# combinations on staright lines and over the curb for  3-4 rounds around the road

# Here is the path to the location of the video files that I have
# recorded in the simulation space !

with open('data_3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
with open('data_2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

with open('data_4/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)
correction = 0.2
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	#Center #left # right
            	for i in range(3):
	                name = batch_sample[i]
	                image_ = cv2.imread(name)
	                # Center
	                if i == 0:
	                	angle = float(batch_sample[3])
	                elif i == 1:
	                	angle = float(batch_sample[3]) + correction
	                elif i == 2:
	                	angle = float(batch_sample[3]) - correction
	                images.append(image_)
	                angles.append(angle)
	                images.append(cv2.flip(image_,1))
	                angles.append(angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x : x/ 255.0 - 0.5 ,input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70, 25),(0, 0))), )
model.add(Convolution2D(24,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64,3,3, activation = 'relu'))
model.add(Convolution2D(48,3,3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam')


# we multiply them by 6 because of data augmentation
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, nb_epoch=2)
model.save('model.h5')