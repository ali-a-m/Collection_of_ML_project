# keras tuner with mnist dataset 

import tensorflow as tf
import keras
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

LOG_DIR = f'{time.time()}'

# get data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# hyperparameters
EPOCHS=50
BATCH_SIZE=64
STEP = 32
MIN_VALUE = 32
MAX_VALUE = 128

# build the CNN Model
def build_model(hp):
	model = Sequential()

	model.add(Conv2D(hp.Int('Input_units', min_value=MIN_VALUE, max_value=MAX_VALUE, step=STEP),
		(3,3),
		padding='same',
		input_shape=x_train.shape[1:],
		activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	for i in range(hp.Int('n_layers', 1, 4)):
		model.add(Conv2D(hp.Int(f'layer_{i}', min_value=MIN_VALUE, max_value=MAX_VALUE, step=STEP),
			(3,3),
			padding='same',
			activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())

	model.add(Dense(10, activation='softmax'))

	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	return model

tuner = RandomSearch(
		build_model,
		objective='val_accuracy',
		max_trials=1,
		executions_per_trial=1,
		directory=LOG_DIR
	)  

# train the model
tuner.search(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

print(tuner.results_summary())