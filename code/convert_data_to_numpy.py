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
from keras import regularizers #import regularizers

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import to_categorical

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import PIL

import pickle

%matplotlib inline

#save and load obj. Ex: dictionary
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
		
#prepare all classes to classify
def prepare_classes(data_dir):
    class_notation = {} #denote classes as number
    total_class = 0 #count number
    for i in os.listdir(data_dir):
        class_notation[i] = total_class
        total_class += 1
    return class_notation, total_class
	
#ONLY RUN ONCE
#CONVERTING ALL THE IMAGES TO NUMPY ARRAY
#BESIDES, IT WILL GENERATE NUMPY ARRAY FOR LABELING
#STORE NUMPY ARRAY AS .NPY FILE TO INPUT FASTER LATER
def prepare_dataset():
    #train dataset directory
    train_dir = './dataset/train/'
  
    #test dataset directory
    test_dir = './dataset/test/'
  
    #get class notation
    class_notation = load_obj('class_notation')
    total_classes = len(class_notation)
  
    #initialize
    X_train_orig = [] #train input
    Y_train_orig = [] #train output
    X_test_orig = [] #test input
    Y_test_orig = [] #test output
  
    print('Load train dataset')
  
    num_files = 0
  
    #load train dataset
    #i: class
    #j: image
    for i in os.listdir(train_dir):
        print('--> ' + i)
    
#         #class_label contains the label of class. Size of class label is number of classes. Initialize with 0
#         class_label = [0] * total_classes
#         class_label[class_notation[i]] = 1
        
        for j in os.listdir(train_dir + i):
            #open image ./train/class_name/image_name
            img = cv2.imread(train_dir + i + '/' + j)
            (b, g, r) = cv2.split(img) 
            img = cv2.merge([r,g,b])
            X_train_orig.append(img)
#             Y_train_orig.append(class_label)
            Y_train_orig.append(class_notation[i])
            num_files += 1
            if num_files % 1000 == 0:
                print(num_files)
        print(num_files)
  
    #convert list to numpy array
    X_train_orig = np.array(X_train_orig, dtype='int32')
    Y_train_orig = np.array(Y_train_orig, dtype='int32')
  
    #save numpy array as .npy format
    np.save('X_train_orig', X_train_orig)
    np.save('Y_train_orig', Y_train_orig)
  
    print('Load test dataset')
  
    #load test dataset
    #i: class
    #j: image
    for i in os.listdir(test_dir):
        print('--> ' + i)
    
#         #class_label contains the label of class. Size of class label is number of classes. Initialize with 0
#         class_label = [0] * total_classes
#         class_label[class_notation[i]] = 1
        
        for j in os.listdir(test_dir + i):
            #open image ./test/class_name/image_name
            img = cv2.imread(test_dir + i + '/' + j)
            (b, g, r) = cv2.split(img) 
            img = cv2.merge([r,g,b])
            X_test_orig.append(img)
#             Y_test_orig.append(class_label)
            Y_test_orig.append(class_notation[i])
            num_files += 1
            if num_files % 1000 == 0:
                print(num_files)
                
        print(num_files)
  
    #convert list to numpy array
    X_test_orig = np.array(X_test_orig, dtype='int32')
    Y_test_orig = np.array(Y_test_orig, dtype='int32')
  
    #save numpy array as .npy format
    np.save('X_test_orig', X_test_orig)
    np.save('Y_test_orig', Y_test_orig)
  
    print('finish:', num_files)
	
#run
prepare_dataset()