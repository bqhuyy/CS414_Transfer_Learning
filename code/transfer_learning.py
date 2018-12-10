#import library
import os,sys #use to go to directory
import math
import numpy as np
import matplotlib.pyplot as plt #use to plot diagram
from PIL import Image #use to load and resize image
import cv2 #read image
from matplotlib.pyplot import imshow

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras import regularizers #import regularizers

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import to_categorical

from keras.preprocessing import image
from keras.callbacks import History 
import PIL

import pickle

%matplotlib inline

from keras.applications.xception import Xception, preprocess_input

#save and load obj. Ex: dictionary
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
		
#LOAD NUMPY ARRAY TO VARIABLES FROM .NPY FILES
def load_dataset():
	global X_train_orig
	global Y_train_orig
	global X_test_orig
	global Y_test_orig
	X_train_orig = np.load('X_train_pre_128.npy') #train input 128x128
	Y_train_orig = np.load('Y_train_pre_128.npy') #train output 128x128
	X_test_orig = np.load('X_test_pre_128.npy') #test input 128x128
	Y_test_orig = np.load('Y_test_pre_128.npy') #test output 128x128

#Loading data
load_dataset()
	
#convert to one hot
Y_train_orig = to_categorical(Y_train_orig)
Y_test_orig = to_categorical(Y_test_orig)

# #Shuffle data
# X_train, Y_train = shuffle_dataset(X_train, Y_train)
# X_test, Y_test = shuffle_dataset(X_test, Y_test)

print ("number of training examples = " + str(X_train_orig.shape[0]))
print ("number of test examples = " + str(X_test_orig.shape[0]))
print ("X_train shape: " + str(X_train_orig.shape))
print ("Y_train shape: " + str(Y_train_orig.shape))
print ("X_test shape: " + str(X_test_orig.shape))
print ("Y_test shape: " + str(Y_test_orig.shape))

# directory
frozen_train = './models/transfer_learning_frozen_128.h5' #freeze model
model_path = './models/transfer_learning_128.h5' #final model
hist_name = 'hist_transfer_learning_128'
acc_name = 'transfer_learning_128_acc.png'
loss_name = 'transfer_learning_128_loss.png'

# load base model
input_tensor = Input(shape=(128, 128, 3)) # change input shape
base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu', name='fc0')(x)
x = Dropout(0.4)(x)
# and a softmax layer -- let's say we have 10 classes
predictions = Dense(10, activation='softmax', name='fc1')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#freeze base model
for layer in base_model.layers:
	layer.trainable = False
	
#train with few epochs with adam lr=0.01
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    frozen_train, verbose=1,
    save_best_only=True)
history = model.fit(x=X_train_orig,y=Y_train_orig,
             validation_data = (X_test_orig, Y_test_orig),
             epochs=5,batch_size=64, shuffle=True,
             callbacks = [cp_callback])
#save history
save_obj(history.history, 'frozen_train')

#unfreeze 1 last block of base_model
for layer in model.layers[:126]:
	layer.trainable = False
for layer in model.layers[126:]:
	layer.trainable = True
	
#recompile model with adam lr=0.01, amsgrad=True
model.compile(optimizer=Adam(lr=0.01, amsgrad=True),loss='categorical_crossentropy',metrics=['accuracy'])
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_path, verbose=1,
    # Save weights, every 5-epochs. period=5
    save_best_only=True)
	
#train
history = model.fit(x=X_train_orig,y=Y_train_orig,
             validation_data = (X_test_orig, Y_test_orig),
             epochs=20,batch_size=64, shuffle=True,
             callbacks = [cp_callback])
save_obj(history.history, hist_name)

#plot loss and accuracy
fig = plt.gcf()
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('./plt/'+acc_name)

# summarize history for loss
fig = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig.savefig('./plt/'+loss_name)