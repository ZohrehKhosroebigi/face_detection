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

np.set_printoptions(threshold=np.nan)
K.set_image_data_format('channels_first')
# create the model from inception model for face images
myFRmodel = Create_model_Images()
myFRmodel.createmdl_img(input_shape=(3, 96, 96))#return myFRmodel.FRmodel

# test Triplet_loss
# it does not need to model from inception model for face images
myloss=Triplet_loss()
print("Strat TF Triplet_loss_________________")
with tf.Session()as sess_test:
    tf.set_random_seed(1)

    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    myloss.triplet_loss(y_true, y_pred, alpha=0.2)
    print("*******" + str(myloss.loss))
    print(myloss.loss)

##########################################
# Loading the pre-trained model
mypre_trained_model =Load_Pre_traind_model()
#mypre_trained_model.compile_model(myFRmodel.FRmodel,optimizer= 'adam',loss=myloss.triplet_loss, metrics = ['accuracy'])
mypre_trained_model.compile_model(myFRmodel.FRmodel, 'adam', myloss.triplet_loss,['accuracy'])

