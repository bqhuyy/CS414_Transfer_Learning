#import library
import os #use to go to directory
import math
import numpy as np
import matplotlib.pyplot as plt #use to plot diagram
from matplotlib.pyplot import imshow

import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.models import load_model
from keras import regularizers #import regularizers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical, plot_model
import pickle

%matplotlib inline

#LOAD NUMPY ARRAY TO VARIABLES FROM .NPY FILES
def load_dataset():
  X_train_orig = np.load('X_train_orig.npy') #train input
  Y_train_orig = np.load('Y_train_orig.npy') #train output
  X_test_orig = np.load('X_test_orig.npy') #test input
  Y_test_orig = np.load('Y_test_orig.npy') #test output
  return X_train_orig, Y_train_orig, X_test_orig, Y_test_orig

#Loading data
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()

#Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

#Reshape
Y_train = to_categorical(Y_train_orig)
Y_test = to_categorical(Y_test_orig)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#save and load obj. Ex: dictionary
def save_obj(obj, name):
  with open(name + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
  with open(name + '.pkl', 'rb') as f:
    return pickle.load(f)
#load class_notation        
class_notation = load_obj('class_notation')
print(class_notation)

#train model (normal_learning_without initialization and dropout)
def CNNModel(input_shape):
  #Define the input placeholder as a tensor with shape input_shape.
  X_input = Input(input_shape)
  
  #Zero-Padding: pads the border of X_input with zeroes
  X = ZeroPadding2D((2,2))(X_input)
  
  #CONV -> BN -> RELU Block applied to X
  X = Conv2D(128, (5,5), strides=(1,1), name='conv0')(X)
  X = BatchNormalization(axis=3, name='bn0')(X)
  X = Activation('relu')(X)
  
  #MAXPOOL
  X = MaxPooling2D((2,2), name='max_pool0', padding='valid')(X)
  
  #CONV -> BN -> RELU Block applied to X
  X = Conv2D(256, (5,5), strides=(1,1), name='conv1')(X)
  X = BatchNormalization(axis=3, name='bn1')(X)
  X = Activation('relu')(X)
  
  #MAXPOOL
  X = MaxPooling2D((2,2), name='max_pool1', padding='valid')(X)
  
  #CONV -> BN -> RELU Block applied to X
  X = Conv2D(512, (5,5), strides=(1,1), name='conv2')(X)
  X = BatchNormalization(axis=3, name='bn2')(X)
  X = Activation('relu')(X)
  
  #MAXPOOL
  X = MaxPooling2D((2,2), name='max_pool2', padding='valid')(X)
  
  
  #FLATTEN X (means convert it to a vector) + 1 FULLYCONNECTED
  X = Flatten()(X)
  X = Dense(1024, activation='relu', name='fc0',kernel_regularizer=regularizers.l2(0.01))(X) #add 1 fc with weight regularizer L2 (l=0.01)
  X = Dropout(0.4)(X)
  X = Dense(10, activation='softmax', name='fc1')(X)
  
  #CREATE MODEL
  model = Model(input = X_input, outputs = X, name='CNNModel')
  
  return model
  
#naming file
model_path = './models/normal_learning.h5'
hist_name = 'hist_normal_learning'
acc_name = 'normal_learning_acc.png'
loss_name = 'normal_learning_loss.png'  

cnnModel = CNNModel(X_train.shape[1:])
cnnModel.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True)

#start training
history = cnnModel.fit(x=X_train,y=Y_train,
             validation_data = (X_test, Y_test),
             epochs=60,batch_size=64, shuffle=True,
             callbacks = [cp_callback])
save_obj(history.history, hist_name)

#draw diagram
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

#then we load the best modal in order to predict manually (modal with lowest val_lost)
cnnModel = load_model('./models/transfer_learning.h5')
def getclassname(num):
  for item in class_notation.items():
    if item[1] == num:
      return item[0]
 
def evaluate_dataset():
  testnum = 13890
  error_test = {}
  for i in range(10): error_test[i] = 0
  dict = class_notation.items()
  
  for i in range(testnum):
    x,y = X_test[i], Y_test[i] 
    x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    res = np.argmax(cnnModel.predict(x))
    y = np.argmax(y)
    if y!=res:
      error_test[y] += 1   
      print('test file \t',i,'should be',getclassname(y),'but prediction is:', getclassname(res))
  return error_test

print(evaluate_dataset())
