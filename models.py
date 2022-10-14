from warnings import filters
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Conv2DTranspose,concatenate,BatchNormalization,Input,Dropout,LeakyReLU,ReLU,MaxPooling2D,UpSampling2D,InputLayer,RNN,Dense,Conv3DTranspose,Conv2D,UpSampling3D,Conv3D,TimeDistributed,BatchNormalization,ConvLSTM2D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import utils
# from tensorflow.data import Dataset
from tensorflow.keras.metrics import mean_squared_error as MSE
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import os,sys
import imageio
import re
import shutil
from time import time
import random
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell, MultiRNNCell
### encoder-decoder without LSTMS 

class VGG_AE(tf.keras.Model):
    def __init__(self):
        super(VGG_AE, self).__init__()
        self.encoder = VGG19(weights='imagenet',include_top =False)
        ### decoder layers
        self.decoder = Sequential([InputLayer(input_shape = (7,7,512)),
                                    UpSampling2D(size= (2,2)),
                                    Conv2DTranspose(input_shape = (14,14,512),filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu',data_format = 'channels_last'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    UpSampling2D(size= (2,2)),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 512,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    UpSampling2D(size= (2,2)),
                                    Conv2DTranspose(filters = 256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 256,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    UpSampling2D(size= (2,2)),
                                    Conv2DTranspose(filters = 128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 128,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    UpSampling2D(size= (2,2)),
                                    Conv2DTranspose(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 64,kernel_size = (3,3),padding = 'same',activation = 'relu'),
                                    Conv2DTranspose(filters = 3,kernel_size = (3,3),padding = 'same',activation = 'relu')])
        self.encoder.trainable = False
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)

class CVAE(tf.keras.Model):
    def __init__(self):
        super(CVAE, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.encoder.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.encoder.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array

class ShrinkCAE(tf.keras.Model):
    def __init__(self):
        super(ShrinkCAE, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(256,448,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(4,4),strides=(4,4)))
        self.encoder.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=16, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=8, kernel_size=(3,3), padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=16,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)
class SimpleCAE(tf.keras.Model):
    def __init__(self):
        super(SimpleCAE, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(224,224,3),filters=16,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(Dense())
        self.encoder.add(MaxPooling2D(pool_size=(8,8),strides=(8,8)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)
class SuperSimpleCAE(tf.keras.Model):
    def __init__(self):
        super(SuperSimpleCAE, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)

class DirectCAE(tf.keras.Model):
    def __init__(self):
        super(DirectCAE, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(256,448,3),filters=8,kernel_size=(3,3),padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(16,16),strides=(16,16)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(16,16)))
        self.decoder.add(Conv2DTranspose(filters=8,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array

class SuperSimpleCAE_KernelIncrease(tf.keras.Model):
    def __init__(self):
        super(SuperSimpleCAE_KernelIncrease, self).__init__()
        # Build Encoder
        self.encoder = Sequential()
        self.encoder.add(Conv2D(input_shape=(224,224,3),filters=32,kernel_size=(9,9),padding="same", activation="relu"))
        self.encoder.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

        # Build Decoder

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=32,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

        self.encoder.trainable = True
        self.decoder.trainable = True

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)

class SENet(tf.keras.Model):
    def __init__(self):
        super(SENet, self).__init__()
        # Build Encoder
        input = Input((224,224,3))
        
        # # x = Dropout(0.5)(x)
        # x = Conv2D(,kernel_size = 1)(x)

        self.decoder = Sequential()
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=256,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=128,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(UpSampling2D(size=(2,2)))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=64,kernel_size=(3,3),padding="same",activation="relu"))
        self.decoder.add(Conv2DTranspose(filters=3,kernel_size = (3,3),padding = 'same',activation = 'relu'))

    def encoder(self,input):
        x = Conv2D(96,kernel_size=(7,7),strides=(2,2),padding='same')(input)
        x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
        x = self.fire_module(x, s1 = 16, e1 = 64, e3 = 64) #2
        x = self.fire_module(x, s1 = 16, e1 = 64, e3 = 64) #3
        x = self.fire_module(x, s1 = 32, e1 = 128, e3 = 128) #4
        x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
        x = self.fire_module(x, s1 = 32, e1 = 128, e3 = 128) #5
        x = self.fire_module(x, s1 = 48, e1 = 192, e3 = 192) #6
        x = self.fire_module(x, s1 = 48, e1 = 192, e3 = 192) #7
        x = self.fire_module(x, s1 = 64, e1 = 256, e3 = 256) #8
        x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
        x = self.fire_module(x, s1 = 64, e1 = 256, e3 = 256) #9
        
    def fire_module(self,x,s1,e1,e3):
        s1x = Conv2D(s1,kernel_size = 1, padding = 'same')(x)
        s1x = ReLU()(s1x)
        e1x = Conv2D(e1,kernel_size = 1, padding = 'same')(s1x)
        e3x = Conv2D(e3,kernel_size = 3, padding = 'same')(s1x)
        x = concatenate([e1x,e3x])
        x = ReLU()(x)
        return x

    def encode(self,inputs):
        img_array = self.encoder(inputs)
        return img_array
    
    def decode(self,inputs):
        img_array = self.decoder(inputs)
        return img_array
        ### output should be (224,224,3)

### with LSTM , same as conflux lstm architecture ###
class VGG_TAE(tf.keras.Model):
    def __init__(self,batch_size,strategy):
        super(VGG_TAE, self).__init__()
        self.num_lstm = 3
        self.chunk_size = 512
        self.rnn_size = 512
        self.seq_length = 20
        self.img_dim = (224,224)
        self.batch_size = batch_size
        self.encoder = VGG19(weights='imagenet',include_top =False)
        ### decoder layers
        self.decoder = Sequential([InputLayer(input_shape = (None,7,7,512)),
                                    UpSampling3D(size= (1,2,2)),
                                    Conv2D(filters = 512,kernel_size = 3,padding = 'same',activation = 'relu',data_format = 'channels_last'),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    Conv2D(filters = 512,kernel_size = 3,padding = 'same',activation = 'relu'),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    Conv2D(filters = 256,kernel_size = 3,padding = 'same',activation = 'relu'),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    Conv2D(filters = 128,kernel_size = 3,padding = 'same',activation = 'relu'),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    Conv2D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu'),
                                    BatchNormalization(),
                                    Conv2D(filters = 3,kernel_size = 3,padding = 'same',activation = 'relu')
                                    ])
        # self.decoder_3d = Conv3D(filters = 3,kernel_size = (3,3,3),padding = 'same',activation = 'relu')
        # with strategy.scope():
        self.W1 = {
            'hidden1': tf.Variable(tf.random.normal([self.chunk_size, self.rnn_size])),
            'output1': tf.Variable(tf.random.normal([self.rnn_size, self.rnn_size]))
        }
        self.B1 = {
                'hidden1': tf.Variable(tf.random.normal([self.rnn_size], mean=1.0)),
                'output1': tf.Variable(tf.random.normal([self.rnn_size]))
            }
        self.lstm_cell = MultiRNNCell([LSTMCell(self.rnn_size, forget_bias=1.0, state_is_tuple=True) for _ in range(self.num_lstm)], state_is_tuple=True)
        self.lstm = RNN(self.lstm_cell,unroll = True,return_sequences = True) # set return sequence to true
        self.fc1 = Dense(7*7*self.rnn_size)
        self.encoder.trainable = False
        
    def call(self,inputs):
        img_h,img_w = self.img_dim
        seq_len = inputs.shape[1]
        inputs = tf.reshape(inputs,(-1,img_h,img_w,3))
        outputs = self.encoder(inputs) ## shape of (seq_length*batch size,7,7,512)
        outputs = tf.reshape(outputs,(self.batch_size,seq_len,-1,outputs.shape[-1])) # (seq_length*batch size,49,512)
        outputs = tf.reshape(tf.math.reduce_mean(outputs,2),(self.batch_size,seq_len,-1)) # (seq_length*batch size,512)
        outputs = tf.nn.relu(tf.matmul(outputs, self.W1['hidden1']) + self.B1['hidden1'])
        # outputs = tf.split(outputs,seq_len,0) # (seq length, batch_size,512)
        # outputs = tf.convert_to_tensor(outputs)
        outputs = self.lstm(outputs) # (becomes batch_size,512)
        outputs = tf.matmul(outputs, self.W1['output1']) + self.B1['output1'] # (seq_length, 512)
        bottleneck = tf.reshape(self.fc1(outputs),(self.batch_size,seq_len,7,7,self.rnn_size))
        decoded_img = self.decoder(bottleneck)
        # decoded_img = tf.reshape(decoded_img,(self.batch_size,seq_len,img_h,img_w,-1))
        # compressed_img = self.decoder_3d(decoded_img)
        return decoded_img


class VGG_LSTMAE(tf.keras.Model):
    def __init__(self,batch_size,strategy):
        super(VGG_LSTMAE, self).__init__()
        self.num_lstm = 3
        self.chunk_size = 512
        self.rnn_size = 512
        self.seq_length = 20
        self.img_dim = (224,224)
        self.batch_size = batch_size
        self.encoder = VGG19(weights='imagenet',include_top =False)
        ### decoder layers
        self.decoder = Sequential([InputLayer(input_shape = (None,7,7,512)),
                                    UpSampling3D(size= (1,2,2)),
                                    ConvLSTM2D(filters = 512,kernel_size = 3,padding = 'same',activation = 'relu',data_format = 'channels_last',return_sequences = True),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    ConvLSTM2D(filters = 512,kernel_size = 3,padding = 'same',activation = 'relu',return_sequences = True),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    ConvLSTM2D(filters = 256,kernel_size = 3,padding = 'same',activation = 'relu',return_sequences = True),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    ConvLSTM2D(filters = 128,kernel_size = 3,padding = 'same',activation = 'relu',return_sequences = True),
                                    BatchNormalization(),
                                    UpSampling3D(size= (1,2,2)),
                                    ConvLSTM2D(filters = 64,kernel_size = 3,padding = 'same',activation = 'relu',return_sequences = True),
                                    BatchNormalization(),
                                    Conv3D(filters = 3,kernel_size = 3,padding = 'same',activation = 'sigmoid')
                                    ])
        # self.decoder_3d = Conv3D(filters = 3,kernel_size = (3,3,3),padding = 'same',activation = 'relu')
        # with strategy.scope():
        self.fc1 = Dense(7*7*self.rnn_size)
        self.encoder.trainable = False
        
    def call(self,inputs):
        img_h,img_w = self.img_dim
        seq_len = inputs.shape[1]
        inputs = tf.reshape(inputs,(-1,img_h,img_w,3))
        outputs = self.encoder(inputs) ## shape of (seq_length*batch size,7,7,512)
        outputs = tf.reshape(outputs,(self.batch_size,seq_len,-1,outputs.shape[-1])) # (seq_length*batch size,49,512)
        outputs = tf.reshape(tf.math.reduce_mean(outputs,2),(self.batch_size,seq_len,-1)) # (seq_length*batch size,512)
        bottleneck = tf.reshape(self.fc1(outputs),(self.batch_size,seq_len,7,7,self.rnn_size))
        decoded_img = self.decoder(bottleneck)
        return decoded_img

