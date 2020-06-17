from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation, Reshape, Permute
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, BatchNormalization
from crfrnn_layer import CrfRnnLayer
import os
import matplotlib.pyplot as plt
import argparse
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.np_utils import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from crfrnn_layer import CrfRnnLayer
from PIL import Image
import numpy as np
import cv2

def dice_coef(y_true, y_pred): 
    smooth = 1. 
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    intersection = K.sum(y_true_f * y_pred_f) 
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) 

def dice_coef_loss(y_true, y_pred): 
    return 1-dice_coef(y_true, y_pred)    

def conv_block(input_tensor, filters, strides, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters[2], kernel_size=1, strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, filters, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
    x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    h = input_tensor.shape[1].value
    w = input_tensor.shape[2].value

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
        x = Conv2D(512, kernel_size=1)(x)
        x = Lambda(lambda x: tf.image.resize_images(x, (h, w)))(x)

        concat_list.append(x)

    return concatenate(concat_list)


def pspnet50(num_classes, input_shape, lr_init, lr_decay):
    img_input = Input(input_shape)
    filters = 64
    
    x = Conv2D(filters, kernel_size=3, strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters*2, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, filters=[filters, filters, filters*4], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[filters, filters, filters*4], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[filters, filters, filters*4], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[filters*2, filters*2, filters*8], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[filters*2, filters*2, filters*8], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[filters*2, filters*2, filters*8], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[filters*2, filters*2, filters*8], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[filters*4, filters*4, filters*16], strides=(1, 1), d_rates=[1, 2, 1])
    x = identity_block(x, filters=[filters*4, filters*4, filters*16], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[filters*4, filters*4, filters*16], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[filters*4, filters*4, filters*16], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[filters*4, filters*4, filters*16], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[filters*4, filters*4, filters*16], d_rates=[1, 2, 1])

    x = conv_block(x, filters=[filters*8, filters*8, filters*32], strides=(1, 1), d_rates=[1, 4, 1])
    x = identity_block(x, filters=[filters*8, filters*8, filters*32], d_rates=[1, 4, 1])
    x = identity_block(x, filters=[filters*8, filters*8, filters*32], d_rates=[1, 4, 1])

    x = pyramid_pooling_block(x, [1, 2, 3, 6])

    x = Conv2D(filters*8, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_classes, kernel_size=1)(x)
    x = Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    x = Activation('sigmoid')(x)

    output = CrfRnnLayer(image_dims=(32, 32),
                         num_classes=num_classes,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])
    
    
    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss=dice_coef_loss,
                  metrics=['accuracy',dice_coef])

    return model

def prepare_image_for_net(img):
    img = img.astype(np.float)
    img /= 255.
    return img

def process_line(line):
    train_path, label_path = line.split(',')
    label_path = label_path.strip('\n')
    train_path = '/input0' + train_path[1:]
    label_path = '/input0' + label_path[1:]
    train = Image.open(train_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label.flags.writeable = True
    train = np.asarray(train)
    label = np.asarray(label)
    label_seg = label
    label_seg[label_seg != 7] = 0
    label_seg[label_seg == 7] = 1
    train = train.reshape((512, 512, 1))
    label_seg = label_seg.reshape((512, 512, 1))
    
    return train, label_seg


def generate_arrays_from_file(path,batch_size):
    while 1:
        f = open(path)
        cnt = 0
        X =[]
        Y =[]
        for line in f:
            x, y = process_line(line)
            X.append(x)
            Y.append(y)
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []
    f.close()
    
 
def train(output="./logs"):
    print('start') 
    score = pspnet50(num_classes=1, input_shape=(512, 512, 1), lr_init=1e-4, lr_decay=5e-4 )
    filepath = "/output/model/v7.model"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    score.fit_generator(generate_arrays_from_file('./meta.csv',batch_size=4),
                                 samples_per_epoch=450//4,nb_epoch=500,validation_data=generate_arrays_from_file('./meta.csv',batch_size=2), validation_steps=150//2, max_q_size=1000,callbacks=callbacks_list, verbose=1, nb_worker=1)
    print("Optimization Finished!")
    
if __name__ == '__main__':
     print('load')
     train()
    
