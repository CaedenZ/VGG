# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 08:04:51 2019

@author: imlab
"""

from __future__ import print_function
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Conv2DTranspose,UpSampling2D,InputLayer
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import utils
from tensorflow.data import Dataset
from tensorflow.keras.metrics import mean_squared_error as MSE
import tensorflow as tf
import tensorflow.keras.backend as K
## configure gpu here
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    try:
        # tf.config.experimental.set_visible_devices(gpus[1], 'GPU') # set to use gpu 1
        tf.config.experimental.set_memory_growth(gpu, True) # set gpu to not consume too much memory
    except RuntimeError as e:
    # Visible devices must be set at program startup
        print(e)

gpu_devices = tf.config.list_logical_devices('GPU')
import numpy as np
import os, sys
import argparse
from scipy import misc
from math import ceil
from PIL import Image
# import cv2
import scipy.io as sio
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
# sys.path.insert(0,'C:/Users/ASUS/Desktop/Machine Leaning Coursera/Project/conflux_lstm/PacmanDQN/videos')
import imageio
import re
import shutil
from time import time
import random
from models import VGG_TAE,VGG_AE,CVAE,SENet,SimpleCAE,SuperSimpleCAE,SuperSimpleCAE_KernelIncrease,ShrinkCAE,DirectCAE


def split_img(img):
    img.resize((184,184))
    img_shape = img.shape[0]
    imgs = np.zeros((img_shape//2,img_shape//2,3,4)) # 4 quadrants
    (h,w) = img.shape[:2]
    (cX,cY) = (w//2,h//2)
    imgs[:,:,:,0] = img[0:cY, 0:cX] # topleft
    imgs[:,:,:,1] = img[0:cY, cX:w]# topright
    imgs[:,:,:,2] = img[cY:h, 0:cX]# btm left
    imgs[:,:,:,3] = img[cY:h, cX:w]# btm right

    return imgs

def sort_name(file):
    regex = re.compile(r'\d+')
    num = regex.findall(file)
    return (int(num[0]))

# def pad_img(img,img_pad):
#     return (cv2.copyMakeBorder(img,img_pad,img_pad,img_pad,img_pad,cv2.BORDER_CONSTANT, (0,0,0)))

def get_cnn_feat(DatasetFolder,model):
    total_dataset = []
    img_s = 184//2
    img_tar = 224
    img_pad = (img_tar - img_s)//2
    for folderName in sorted(os.listdir(DatasetFolder),key = sort_name): # iterating over 4 quadrants
        # dataset_feat = []
        ep_folder = DatasetFolder + '/' + folderName
        frame_no=0
        vid_feats = np.zeros((len(os.listdir(ep_folder))//2,512,4))
        for img_file in sorted(os.listdir(ep_folder),key = sort_name): # iterating over images
            if img_file.endswith('.png'):
                img = ep_folder + '/' + img_file

                img_data = imageio.imread(img)
                splitted_img = split_img(img_data) # take img and split to 4 quadrants
                for quad in range(splitted_img.shape[-1]): # process each quad
                    quad_img = splitted_img[:,:,:,quad]
                    # padded_img = pad_img(quad_img,img_pad)
                    img_data = np.expand_dims(quad_img, axis=0)
                    img_data = preprocess_input(img_data)

                    vgg19_feature = model.predict(img_data)
                    blob1 = vgg19_feature.reshape([49,512])   
                    arr = np.array(blob1)
                    abcd= arr.mean(0)
                    vgg19_feature = abcd.reshape([1,512])                    
                    features = np.array(vgg19_feature)
                    vid_feats[frame_no,:,quad]= features # append features into vid_feats of shape (ep,dim,quad)

    
                frame_no +=1
                if frame_no == 2:
                    break
#                 if frame_no % 15 ==0: # collect 15 frames to form a video 
# #                        print(frame_no)
#                     # aa = np.asarray(vid_feat)
                    # dataset_feat.append(aa)
                    # vid_feat=[]
        total_dataset.append(vid_feats)
        break
    return total_dataset


def combine_img_to_file(DatasetFolder):
    target_dir = 'img_set'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    ep_count = 1
    for folderName in sorted(os.listdir(DatasetFolder),key = sort_name):
        ep_folder = DatasetFolder + '/' + folderName
        for img_file in sorted(os.listdir(ep_folder),key = sort_name): # iterating over images
            if img_file.endswith('.png'):
                img = ep_folder + '/' + img_file
                if os.path.exists(target_dir + '/' + img_file):
                    img_file = f'{ep_count}_' + img_file 
                shutil.copyfile(img,target_dir + '/' + img_file)
        ep_count += 1
    return ep_count

def compute_cnn_loss(model,x):
    encoded_img = model.encode(x)
    x_logits = model.decode(encoded_img)
    cross_loss = tf.keras.losses.MeanSquaredError()
    loss = cross_loss(x,x_logits)
    # loss = MSE(x,x_logits)
    return loss

def compute_loss(model,x,y,strategy = None):
    pred_img = model(x)
    cross_loss = tf.keras.losses.MeanSquaredError()
    # cross_loss = tf.keras.losses.BinaryCrossentropy()
    loss = cross_loss(y,pred_img)
    # loss = MSE(x,x_logits)
    return loss

def train_func(model,x,optimizer,strategy=None): ## training for reconstruction of imgs
    with tf.GradientTape() as tape:
        loss = compute_cnn_loss(model,x)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss

def pred_train_func(model,x,y,optimizer,strategy=None): ## training for prediction of next img
    with tf.GradientTape() as tape:
        loss = compute_loss(model,x,y,strategy)
    grads = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    return loss

### training function
def run_training(train_ds,test_ds,epochs,test_sample,img_dict,batch_size,img_dim,strategy=None):
    # with strategy.scope(): ## just need set initalization of model and optimizer inside
        ## model
    cnn_model = DirectCAE() ## VGG_AE for reconstruction of img, not temporal
    optimizer = tf.keras.optimizers.Adam(lr = 1e-4) # lr = 0.001
    generate_save_img(cnn_model,0,test_sample,batch_size,img_dim,img_dict)
    ### training
    trainable_count = np.sum([K.count_params(w) for w in cnn_model.trainable_weights])
    print (f'Total number of trainable params : {np.round(trainable_count/1e6,2)}M')
    print (f'Total training batch: {len(train_ds)}')
    print (f'Total test batch: {len(test_ds)}')
    img_h,img_w = img_dim
    trainloss = []
    testloss = []
    for epoch in range(epochs):
        st_t = time()
        for i,train_batch in enumerate(train_ds):
            batch_data = tf.reshape(train_batch,(-1,img_h,img_w,3))
            train_loss = train_func(cnn_model,batch_data,optimizer)
            print (f'epoch: {epoch+1} , batch: {i+1} loss : {train_loss}')

        for test_batch in test_ds:
            test_batch_data = tf.reshape(test_batch,(-1,img_h,img_w,3))
            test_loss = compute_cnn_loss(cnn_model,test_batch_data)
        trainloss.append(np.round(train_loss.numpy(),2))
        testloss.append(np.round(test_loss.numpy(),2))
        print (f'epoch: {epoch+1} , train loss : {np.round(train_loss.numpy(),2)}, test loss: {np.round(test_loss.numpy(),2)}, time taken : {np.round(time()-st_t,2)}s')
        generate_save_img(cnn_model,epoch+1,test_sample,batch_size,img_dim,img_dict)
    plt.clf()
    fig = plt.figure()
    plt.plot(trainloss)
    plt.plot(testloss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')

def crop_img(img):
    o_h,o_w = img.shape[-2],img.shape[-1]
    cropped_img = img[:0.9*o_h,:]
    return cropped_img

def generate_save_img(model,epoch,test_sample,batch_size,img_dim,img_dict):
    e_img = model.encode(test_sample)
    d_img = model.decode(e_img)
    random_idx = random.randint(0,batch_size-1)

    fig = plt.figure()
    f,ax = plt.subplots(1,2,figsize=(10,6))
    ax[0].imshow(tf.cast(test_sample[random_idx],tf.int32))
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(tf.cast(d_img[random_idx],tf.int32))
    ax[1].axis('off')
    ax[1].set_title('Reconstructed Image')

    plt.savefig(f'{img_dict}/image_{epoch}.png')

def sort_name(file):
    regex = re.compile(r'\d+')
    num = regex.findall(file)
    return (int(num[0]))

# generate test img
def plot_img(img_ds):
    for imgs in img_ds.take(1):
        img = imgs[0]
        num_r = 5
        num_c = 4
        img_count = 0
        f,ax = plt.subplots(5,4,figsize = (20,12))
        for i in range(num_r):
            for j in range(num_c):
                ax[i,j].imshow(tf.cast(img[img_count],tf.int32))
                ax[i,j].axis('off')
                ax[i,j].set_title(f'timestep {img_count+1}')
                img_count +=1
        plt.savefig('test_sequence_img.png')

def generate_seq_img(model,epoch,test_sample,batch_size,img_dim,img_dict):
    pred_len = 10
    img_h,img_w = img_dim
    random_idx = random.randint(0,batch_size-1)
    img_data = test_sample # randomly sample 1 sequence of img
    sample_data_x = img_data[:,:pred_len]
    s_data_x = tf.identity(sample_data_x)
    sample_data_y = img_data[random_idx,pred_len:]
    for i in range(pred_len):
        # sample_data_x = tf.keras.preprocessing.sequence.pad_sequences(sample_data_x,maxlen =19)
        pred_img = model(sample_data_x)
        # pred_img = np.expand_dims(model(sample_data_x),axis = 0)
        pred_img =  np.expand_dims(pred_img[:,-1],axis = 1) # take last predicted img.
        sample_data_x = np.concatenate((sample_data_x,pred_img),axis = 1) # feed in autoregressively
        sample_data_x = sample_data_x[:,-10:]
    
    fig = plt.figure()
    f,axes = plt.subplots(2,10,figsize=(20,4))
    for idx, ax in enumerate(axes[0]):
        ax.imshow(tf.cast(np.squeeze(sample_data_y[idx]),tf.int32),cmap='gray')
        ax.axis('off')
        ax.set_title(f'Original - {idx+11}')

    new_frames = sample_data_x[random_idx]
    for idx, ax in enumerate(axes[1]):
        ax.imshow(tf.cast(np.squeeze(new_frames[idx]),tf.int32), cmap="gray")
        ax.set_title(f"Frame {idx + 11}")
        ax.axis("off")

    plt.savefig(f'{img_dict}/image_{epoch}.png')


## configure strategy for gpu

# strategy = tf.distribute.MirroredStrategy(gpu_devices)
strategy = None

# for i, layer in enumerate(cnn_model.encoder.layers):
#    print(i, layer.name)
# cnn_model.encoder.summary()
img_h,img_w = 256 ,448
img_dim = (img_h,img_w)
# batch_size = 32 * strategy.num_replicas_in_sync
seq_length = 20
batch_size = 16 
num_epochs = 10
train_size = 0.8

result_img_dict = 'test' ## set your results folder here
if not os.path.exists(result_img_dict):
    os.makedirs(result_img_dict)

# img_folder = 'medium_grid_img'
img_folder = 'blackened' # set your image dataset folder here

## change file name
# for files in sorted(os.listdir(img_folder),key = sort_name):
#     for file_n in sorted(os.listdir(img_folder + '/' + files),key = sort_name):
#         regex = re.compile(r'\d+')
#         num = regex.findall(file_n)
#         fil_num =  int(num[0])
#         if fil_num <10 and len(str(file_num)) < 2:
#             os.rename(os.path.join(img_folder,files,file_n),os.path.join(img_folder,files,"0"+file_n))

filename_ds = Dataset.list_files(f"{img_folder}/*.png") # if folder contains all images, no subdirectories 

# img_ds = utils.image_dataset_from_directory(img_folder,labels = None,batch_size = seq_length,shuffle = False,image_size =(301,476))

### preprocess img
img_ds = filename_ds.map(lambda x: (tf.io.decode_image(tf.io.read_file(x), expand_animations = False))) ## decode filename into image

img_ds = img_ds.map(lambda x : (tf.image.crop_and_resize(tf.expand_dims(x,0),[[0.0,0.0,0.9,1.0]], [0], crop_size=(img_h,img_w)))) # crop out score and resize to (224,224) ---> input size into vgg19

img_ds = img_ds.map(lambda x : tf.squeeze(x)) # remove added dim

# img_ds = img_ds.batch(batch_size).shuffle(len(img_ds))
img_ds = img_ds.shuffle(len(img_ds)).batch(batch_size,drop_remainder = True) # shuffle and batch
# img_ds = img_ds.batch(batch_size,drop_remainder =True)

## plot out 1 img to sample
# plot_img(img_ds)

## split into train and test
train_ds = img_ds.take(ceil(train_size*len(img_ds))) 
test_ds = img_ds.skip(ceil(train_size*len(img_ds)))

## take a random test sample to observe performance as training goes
for test_b in test_ds.take(1):
    test_sample = test_b
    ## generate test img to see
    # fig = plt.figure()
    # plt.imshow(tf.cast(test_sample[0],tf.int32))
    # plt.axis('off')
    # plt.savefig('test_img.png')

## run training
# tf.device(gpus[2])
# os.environ["CUDA_VISIBLE_DEVICES"]="2"
run_training(train_ds,test_ds,num_epochs,test_sample,result_img_dict,batch_size,img_dim,strategy)
print('Done')





