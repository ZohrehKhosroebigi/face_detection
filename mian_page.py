"""
keras version = 2.0.7
tensorflow version = 1.2.1
This network uses 96x96 dimensional RGB images as its input, a tensor of shape  (m,nC,nH,nW)=(m,3,96,96)(m,nC,nH,nW)=(m,3,96,96)
It outputs a matrix of shape  (m,128) that encodes each input face image into a 128-dimensional vector
"""
from Load_Pre_traind_model import Load_Pre_traind_model
from create_model_for_images import Create_model_Images
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import inception_blocks_v2
# import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import fr_utils
from Triplet_loss import Triplet_loss
from Create_database_128 import Create_database_128
from verify import Verify
from Recognition import Recognition
from User_page import User_page

np.set_printoptions(threshold=np.nan)
K.set_image_data_format('channels_first')
# create the model from inception model for face images
myFRmodel = Create_model_Images()
myFRmodel.createmdl_img(input_shape=(3, 96, 96))#return myFRmodel.FRmodel

# test Triplet_loss
# it does not need to model from inception model for face images
myloss=Triplet_loss()

# Loading the pre-trained model
mypre_trained_model =Load_Pre_traind_model()
#mypre_trained_model.compile_model(myFRmodel.FRmodel,optimizer= 'adam',loss=myloss.triplet_loss, metrics = ['accuracy'])
mypre_trained_model.compile_model(myFRmodel.FRmodel, 'adam', myloss.triplet_loss,['accuracy'])


#to build the database. This database maps each person's name to a 128-dimensional encoding of their face.
mydatabase=Create_database_128()
mydatabase.create_data_128(myFRmodel.FRmodel)

#############
myuser=User_page()
myuser.user_page
####################

#applying the model for face verification
myverification=Verify()
myverification.verify("images/camera_0.jpg", "younes", mydatabase.database, myFRmodel.FRmodel)
print(myverification)

#applying the model for face recognition
myrecognition=Recognition()
myrecognition.recognition(myuser.img_path, mydatabase.database, myFRmodel.FRmodel)
print(myrecognition)
