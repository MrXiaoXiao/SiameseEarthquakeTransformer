from __future__ import print_function
from __future__ import division
import keras
from keras import backend as K
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.layers import Input, InputLayer, Lambda, Dense, Flatten, Conv2D, BatchNormalization
from keras.layers import UpSampling2D, Cropping2D, Conv2DTranspose, Concatenate, Activation 
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import csv
import h5py
import time
from os import listdir
import os
import shutil
from EqT_utils import DataGeneratorPrediction, picker, generate_arrays_from_file
from EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
from tqdm import tqdm
from datetime import datetime, timedelta
import multiprocessing
import contextlib
import sys
import warnings
from scipy import signal
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
from keras_radam import RAdam
from keras.initializers import Constant
deprecation._PRINT_DEPRECATION_WARNINGS = False

def feature_map_corr_func(inputs):
    # get batch size for split tensor
    batch_size = 1

    # template
    input_template = inputs[0]
    Ht, Wt, Bt, Ct = tf.unstack(tf.shape(input_template))
    input_template = tf.transpose(input_template,perm=[1,2,0,3])
    input_template = tf.reshape(input_template, (Ht, Wt, Bt*Ct, 1))
    
    # search
    input_search = inputs[1]
    Hs, Ws, Bs, Cs = tf.unstack(tf.shape(input_search))
    input_search = tf.transpose(input_search, perm=[1,2,0,3])
    input_search = tf.reshape(input_search, (1, Hs, Ws, Bs*Cs))

    # feature correlation
    feature_corr = tf.nn.depthwise_conv2d(input_search,
                                        input_template,
                                        strides=[1,1,1,1],
                                        padding='SAME')
    
    #feature_corr = tf.concat(tf.split(feature_corr,batch_size,axis=3),axis=0)
    feature_corr = tf.transpose(feature_corr,perm=[0,2,1,3])
    return feature_corr

def feature_map_corr_layer(name='feature_map_corr',output_shape=(47,1,16)):
    return Lambda(feature_map_corr_func, output_shape=output_shape, name=name)

def build_corr_model(cfgs, EqT_model):
    """
    build cross correlation model
    """
    encoded_list = cfgs['Model']['RSRN_Encoded_list']

    output_dict = dict()
    output_list = []

    for encoded_name in encoded_list:
        output_dict[encoded_name] =  EqT_model.get_layer(encoded_name).output
        output_list.append(output_dict[encoded_name])

    model_encoded = Model(inputs = EqT_model.input, outputs = output_list)

    # S-EqT-RSRN
    # length list
    encoded_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    encoded_channels = cfgs['Model']['RSRN_Encoded_channels']

    # define inputs
    S_EqT_Input_dict = dict()
    S_EqT_Input_list = []
    for idx, encoded_name in enumerate(encoded_list):
        S_EqT_Input_dict[encoded_name+'_Template'] = Input(shape=[None,1,int(encoded_channels[idx])],name=encoded_name+'_Template')
        S_EqT_Input_dict[encoded_name+'_Search'] = Input(shape=[int(encoded_lengths[idx]),1,int(encoded_channels[idx])],name=encoded_name+'_Search')
        S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Template'])
        S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Search'])

    # define correlation results
    concate_with_ori = int(cfgs['Model']['Concate_with_ori'])
    feature_corr_dict = dict()

    S_EqT_Output_list = []
    for idx, encoded_name in enumerate(encoded_list):
        feature_corr_dict[encoded_name+'_corr']  = feature_map_corr_layer(encoded_name+'_corr',(int(encoded_lengths[idx]),1,int(encoded_channels[idx])))([S_EqT_Input_dict[encoded_name+'_Template'],S_EqT_Input_dict[encoded_name+'_Search'] ])
        S_EqT_Output_list.append(feature_corr_dict[encoded_name+'_corr'])
    model_corr = Model(inputs = S_EqT_Input_list, outputs = S_EqT_Output_list)

    return model_corr

def S_EqT_Concate_RSRN_Model(cfgs):
    """
    Concate RSRN & 
    """
    print('Start loading EqT model...')
    # EqT encoded model
    EqT_model_path = cfgs['Model']['EqT_model_path']
    EqT_model = load_model(EqT_model_path, 
                   custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                   'FeedForward': FeedForward,
                                   'LayerNormalization': LayerNormalization, 
                                   'f1': f1                                                                            
                                    })
    
    print('Start building EqT encoder model...')
    encoded_list = cfgs['Model']['RSRN_Encoded_list']

    output_dict = dict()
    output_list = []
    
    for encoded_name in encoded_list:
        output_dict[encoded_name] =  EqT_model.get_layer(encoded_name).output
        output_list.append(output_dict[encoded_name])
    
    if_encoder_concate = int(cfgs['Model']['Encoder_concate'])

    if if_encoder_concate == 1:
        encoder_encoded_list = cfgs['Model']['Encoder_concate_list']
        encoder_encoded_lengths = cfgs['Model']['Encoder_concate_lengths']
        encoder_encoded_channels = cfgs['Model']['Encoder_concate_channels']
        
        for encoded_name in encoder_encoded_list:
            output_dict[encoded_name] =  EqT_model.get_layer(encoded_name).output
            output_list.append(output_dict[encoded_name])
    
    model_encoded = Model(inputs = EqT_model.input, outputs = output_list)
    # EqT encoded model END
    print('Start building Siamese EqT model...')
    # S-EqT-RSRN
    # length list
    encoded_lengths = cfgs['Model']['RSRN_Encoded_lengths']
    encoded_channels = cfgs['Model']['RSRN_Encoded_channels']
    # define inputs
    S_EqT_Input_dict = dict()
    S_EqT_Input_list = []
    for idx, encoded_name in enumerate(encoded_list):
        S_EqT_Input_dict[encoded_name+'_Template'] =  Input(shape=[None,1,int(encoded_channels[idx])],name=encoded_name+'_Template')
        S_EqT_Input_dict[encoded_name+'_Search'] = Input(shape=[int(encoded_lengths[idx]),1,int(encoded_channels[idx])],name=encoded_name+'_Search')
        S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Template'])
        S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Search'])
    
    if if_encoder_concate == 1:
        for idx, encoded_name in enumerate(encoder_encoded_list):
            S_EqT_Input_dict[encoded_name+'_Template'] =  Input(shape=[None,1,int(encoder_encoded_channels[idx])],name=encoded_name+'_Template')
            S_EqT_Input_dict[encoded_name+'_Search'] = Input(shape=[int(encoder_encoded_lengths[idx]),1,int(encoder_encoded_channels[idx])],name=encoded_name+'_Search')
            S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Template'])
            S_EqT_Input_list.append(S_EqT_Input_dict[encoded_name+'_Search'])
            
    print('Start building correlation layers...')
    # define correlation results
    concate_with_ori = int(cfgs['Model']['Concate_with_ori'])
    feature_corr_dict = dict()
    for idx, encoded_name in enumerate(encoded_list):
        if concate_with_ori == 1:
            corr_res = feature_map_corr_layer(encoded_name+'_corr',
                                                (int(encoded_lengths[idx]),
                                                1,int(encoded_channels[idx])))([S_EqT_Input_dict[encoded_name+'_Template'],S_EqT_Input_dict[encoded_name+'_Search'] ])
            #corr_res = BatchNormalization()(corr_res)
            feature_corr_dict[encoded_name+'_corr'] = Concatenate(axis=-1)([corr_res,S_EqT_Input_dict[encoded_name+'_Search']])
        else:
            feature_corr_dict[encoded_name+'_corr'] = feature_map_corr_layer(encoded_name+'_corr',
                                                (int(encoded_lengths[idx]),
                                                1,int(encoded_channels[idx])))([S_EqT_Input_dict[encoded_name+'_Template'],S_EqT_Input_dict[encoded_name+'_Search'] ])
            #feature_corr_dict[encoded_name+'_corr'] = BatchNormalization()(corr_res)
    
    if if_encoder_concate == 1:
        encoder_concate_list = list()
        for idx, encoded_name in enumerate(encoder_encoded_list):
            if concate_with_ori == 1:
                corr_res = feature_map_corr_layer(encoded_name+'_corr',
                                                    (int(encoder_encoded_lengths[idx]),
                                                    1,int(encoder_encoded_channels[idx])))([S_EqT_Input_dict[encoded_name+'_Template'],S_EqT_Input_dict[encoded_name+'_Search'] ])
                #corr_res = BatchNormalization()(corr_res)
                feature_corr_dict[encoded_name+'_corr'] = Concatenate(axis=-1)([corr_res,S_EqT_Input_dict[encoded_name+'_Search']])
            else:
                feature_corr_dict[encoded_name+'_corr']  = feature_map_corr_layer(encoded_name+'_corr',
                                                    (int(encoder_encoded_lengths[idx]),
                                                    1,int(encoder_encoded_channels[idx])))([S_EqT_Input_dict[encoded_name+'_Template'],S_EqT_Input_dict[encoded_name+'_Search'] ])

            encoder_concate_list.append(feature_corr_dict[encoded_name+'_corr'])
    
        encoder_list_concate_final = Concatenate(axis=-1)(encoder_concate_list)
        side_conv = Conv2D(512,(3,1),padding='same')(encoder_list_concate_final)
        side_conv = Conv2D(256,(3,1),padding='same')(side_conv)
        side_conv = Conv2D(128,(3,1),padding='same')(side_conv)
        side_conv = Conv2DTranspose(128, (2,1), strides=(2,1), padding='same', activation=None)(side_conv)
        side_conv = Conv2D(64,(3,1),padding='same')(side_conv)
        side_conv = Conv2D(64,(3,1),padding='same')(side_conv)

    # concate by group
    print('Start sideoutput & residual layers...')
    # define side outputs
    # sideoutput_before_activation = dict()
    sideoutput_dict = dict()
    side_residual_dict = dict()
    sideoutput_upscales = cfgs['Model']['Sideoutput_Upscales']
    sideoutput_croppings = cfgs['Model']['Sideoutput_Croppings']
    residual_upscales =  cfgs['Model']['Residual_Upscales']
    residual_croppings = cfgs['Model']['Residual_Croppings']
    output_list_siamese = list()
    stage_output_list_siamese = list()
    
    for idx, encoded_name in enumerate(encoded_list):
        # if bottom layer
        if idx == 0:
            if if_encoder_concate == 1:
                side_concate = Concatenate(axis=-1)([feature_corr_dict[encoded_name+'_corr'],side_conv])
                side_conv = Conv2D(32,(3,1),padding='same')(side_concate)
            else:
                side_conv = Conv2D(32,(3,1),padding='same')(feature_corr_dict[encoded_name+'_corr'])
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(16,(1,1),padding='same')(side_conv)

            side_residual_dict[encoded_name+'_resiudal'] = Lambda(lambda x: x[:,:,:,0:8]) (side_conv)
            classifier = Lambda(lambda x: x[:,:,:,8:16])(side_conv)
            # get upscale
            upscale = int(sideoutput_upscales[idx])
            # cropping?
            cropping = int(sideoutput_croppings[idx])
            if cropping == 0:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same', activation=None)(classifier)
            else:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same', activation=None)(classifier)
                classifier = Cropping2D(cropping=((cropping,cropping),(0,0)))(classifier)
            stage_output_list_siamese.append(classifier)
            sideoutput_dict[encoded_name+'_sideoutput'] = Activation('sigmoid')(classifier)
            output_list_siamese.append(sideoutput_dict[encoded_name+'_sideoutput'])

        # if top layer
        elif idx == len(encoded_list) - 1:
            side_conv = Conv2D(32,(3,1),padding='same')(feature_corr_dict[encoded_name+'_corr'])
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(16,(1,1),padding='same')(side_conv)
            # combine residual and res
            residual_name = encoded_list[idx-1] + '_resiudal'
            residual = Conv2DTranspose(8,(3,1),strides=(int(residual_upscales[idx-1]),1),padding='same')(side_residual_dict[residual_name])
            res_cropping = int(residual_croppings[idx])
            if res_cropping == 0:
                pass
            else:
                residual = Cropping2D(cropping=((res_cropping,res_cropping),(0,0)))(residual)

            side_concat = Concatenate(axis=-1)([side_conv, residual])
            side_conv = Conv2D(8,(1,1),padding='same')(side_concat)
           # get upscale
            upscale = int(sideoutput_upscales[idx])
            # cropping?
            cropping = int(sideoutput_croppings[idx])
            if cropping == 0:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same')(side_conv)
            else:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same')(side_conv)
                classifier = Cropping2D(cropping=((cropping,cropping),(0,0)))(classifier)
            stage_output_list_siamese.append(classifier)
            sideoutput_dict[encoded_name+'_sideoutput'] = Activation('sigmoid')(classifier)
            output_list_siamese.append(sideoutput_dict[encoded_name+'_sideoutput'])
        # else middle layer
        else:
            side_conv = Conv2D(32,(3,1),padding='same')(feature_corr_dict[encoded_name+'_corr'])
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(32,(3,1),padding='same')(side_conv)
            side_conv = Conv2D(16,(1,1),padding='same')(side_conv)
            # combine residual and res
            residual_name = encoded_list[idx-1] + '_resiudal'
            residual = Conv2DTranspose(8,(3,1),strides=(int(residual_upscales[idx-1]),1),padding='same')(side_residual_dict[residual_name])
            
            res_cropping = int(residual_croppings[idx])
            if res_cropping == 0:
                pass
            else:
                residual = Cropping2D(cropping=((res_cropping,res_cropping),(0,0)))(residual)

            side_concat = Concatenate(axis=-1)([side_conv, residual])
            side_conv = Conv2D(16,(1,1),padding='same')(side_concat)

            side_residual_dict[encoded_name+'_resiudal'] = Lambda(lambda x: x[:,:,:,0:8]) (side_conv)
            
            classifier = Lambda(lambda x: x[:,:,:8:16])(side_conv)
            # get upscale
            upscale = int(sideoutput_upscales[idx])
            # cropping?
            cropping = int(sideoutput_croppings[idx])
            if cropping == 0:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same')(classifier)
            else:
                classifier = Conv2DTranspose(1, (1,1), strides=(upscale,1), padding='same')(classifier)
                classifier = Cropping2D(cropping=((cropping,cropping),(0,0)))(classifier)
            
            stage_output_list_siamese.append(classifier)
            sideoutput_dict[encoded_name+'_sideoutput'] = Activation('sigmoid')(classifier)
            output_list_siamese.append(sideoutput_dict[encoded_name+'_sideoutput'])
    # fuse all
    fuse = Concatenate(axis=-1)(stage_output_list_siamese)
    f_conv_start_weight = 1.0/(float(len(stage_output_list_siamese)))
    f_conv_init = Constant(f_conv_start_weight)
    fuse = Conv2D(1, (1,1), padding='same', activation='sigmoid',kernel_initializer=f_conv_init)(fuse)
    # build and complie model 
    output_list_siamese.append(fuse)
    model_siamese = Model(inputs = S_EqT_Input_list, 
                                outputs = output_list_siamese)
    
    loss_list = []
    loss_weights = []
    loss_weights_cfgs = cfgs['Model']['Loss_weights']
    for l_w in loss_weights_cfgs:
        loss_weights.append(float(l_w))
    """
    model_siamese.compile(loss='binary_crossentropy',
                            optimizer=RAdam(0.0003))
    """
    model_siamese.compile(loss='binary_crossentropy',
                            optimizer='adam')
    return model_encoded, model_siamese, EqT_model