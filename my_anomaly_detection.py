'''
Python module for my own Anomaly Detection.


Classes:
sampling2
CustomModelCheckpoint



Functions:
vae_vsam
vae
get_callbacks






Note:
- turn more stuff into classes...
- cleanup

'''
import warnings
import pickle
#general 
import sys
import numpy as np
import pandas as pd
import math
import os
from time import time
import datetime

# RNN, deep leraning ect.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,GlobalMaxPool1D
from tensorflow.keras.layers import Flatten,Concatenate,Attention,Add,BatchNormalization
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, SimpleRNN, GRU, TimeDistributed, ConvLSTM2D, RNN,Conv1D
from tensorflow.keras.layers import RepeatVector, Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import TimeDistributed, Attention
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from keras import backend as K
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations

from keras import backend as K


class Sampling2(keras.layers.Layer):
    '''
        sampling layer...
            t
        Note:
            try and change the mean and std in epsilon. (get a bachlor student to it.)
            
        author:
            Kristian S
    '''
    def call(self,inputs):
        mean,log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
        return mean + K.random_normal(tf.shape(log_var))*K.exp(log_var/2) * epsilon    
    
class MyCustomWarning(UserWarning):
    pass

def vae_vsam(seq_length = 310,
           features = 240,
           coding_size = 50,
           with_gpu=False,
           blstm = [200,100],
           lr_init =0.001,
           general_loss = 'KLDivergence',
           optimizer  = 'Nadam',
           verbose=0,
           gpu_single=True,
           gpu_single_number = '1',
           blstm_regu= 0.00001,
           recurrent_dropout=0.6,
           dropout= 0.6,
           log_likelohood_regul = 0.00001,
           DL_regu= 0.001,
           custom_loss_regularisation = 0.01
          ):
    '''
    kl_divergence,cosine_similarity
    Function to make a Variational Autoencoder with variational self-attention mechanism.
    -------------------------------------------
    This function makes a Variational Autoencoder with Bidirectional Long Short Term Memory. This VAE can be used to:
    1) Detect anomalies in data (data similar to the type used for training)
    2) Create synthetic data similar to the data used for training.
    
    The VAE is inherently probibalistic.Here, a Gaussian VAE is made. For mode information on VAE, see the thesis.
    
    Firstly, an ENCODER is made. Then, the latent vector is computed (also called bottleneck). 
    Then, the DECODER is made. This decoder takes the latent vector and tries to compute the original sequence. 
    
    This is all combines in the model call vae.
    
    The input parameters are all saved in a folder designated for this model. These paramters should thus be used to create the model at a later data.
    The vae weights and history will similary be saved in this folder. 
    The history can be accesed as a dataframe and consists of the loss and acc.
    The weights can be loaded on the model at a later date
    
    Since this model uses costum layers, e.g. the Sampling() layer, the model cant be saved as a Keras model.
    Instead, the weights are saved. Consequently, to recontruct the model at a later date do the following:
        1) Re-create the model using the input parameters in the model folder.
        2) load the weigths from the model folder on the newly created model.
        3) Use the model with the corect weights for whatever you want.
        
    --------------------------------------------
    Input:
           seq_length = samples_scaled_padded_reshaped.shape[1],
           features = samples_scaled_padded_reshaped.shape[2],
           coding_size = 50,
           with_gpu=False,
           blstm = [200,100],
           lr_init =0.001,
           general_loss = 'cosine_similarity',
           optimizer  = 'Nadam',
           verbose=0
    Output:
           vae_model: The variational autoencoder
           encoder: The Encoder
           decode: The Decoder
           
    Example: 
           vae_model,encoder,decoder= cl_vae()
    
    Author:
           Kristian Soerensen
           kristian.sorensen@hotmail.com
           November 2020.
    Note:           
    '''
    print("Warning: Only the weights and history can be saved.Due to the costum layer 'Sampling', model can no be saved. Therefore, when loading the model, save the inputs to create the omdel anew and then load weight ontop of it.\n")
    
    name = f"VAE_VSAM_model_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    path = name
    os.makedirs(path, exist_ok=True) 
    if with_gpu==True:
        if with_gpu==True and gpu_single==False:
            #strategy = tf.distribute.MirroredStrategy()
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy().scope()
        if with_gpu==True and gpu_single==True:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            #tf.config.experimental.set_memory_growth(gpus, True)
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(gpus[int(gpu_single_number)], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=23000)])
                except RuntimeError as e:
                    print(e)
                    
            strategy = tf.device(f"/GPU:{gpu_single_number}")
            
        if verbose>0:
            print('Number of devices: \n{}'.format(strategy.num_replicas_in_sync))
            
            
        with strategy:
            #inputs for encoder and decoder
            encoder_inputs = Input(shape=([seq_length,features]),name='encoder_input')
            decoder_inputs = Input(shape=(coding_size*2),name='decoder_input')
            #masking layer for 0 values
            #masking = Masking(mask_value=0,input_shape=(None,features),name='encoder_masking')(encoder_inputs)
            #############################################################
            ########################## ENCODER ########################
            ############################################################
            #Encoder layers
            z1 = Bidirectional(LSTM(blstm[0],
                                    batch_input_shape=(None, seq_length, features),
                                    recurrent_dropout=recurrent_dropout, 
                                    kernel_regularizer=l2(blstm_regu), 
                                    recurrent_regularizer=l2(blstm_regu), 
                                    bias_regularizer=l2(blstm_regu),
                                    name='encoder_blstm1',
                                    return_sequences= True))(encoder_inputs)
            #file:///C:/Users/krist/AppData/Local/Temp/1602.02282.pdf
            z = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(z1)
            z = Dropout(dropout)(z)
            
            z = Bidirectional(LSTM(blstm[1],
                                   recurrent_dropout=recurrent_dropout, 
                                   kernel_regularizer=l2(blstm_regu), 
                                   recurrent_regularizer=l2(blstm_regu), 
                                   bias_regularizer=l2(blstm_regu),
                                   name='encoder_blstm2'))(z)
            z = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(z)
            
            z = Dropout(dropout)(z)
            initializer = tf.keras.initializers.Zeros()
            
            z_mean = Dense(coding_size,kernel_initializer=initializer,name='encoder_Z_mean')(z)
            z_variance = Dense(coding_size,kernel_initializer=initializer,name='encoder_Z_variance',activation=activations.softplus)(z)
            
            latent_vector = Sampling2()([z_mean,z_variance])
            #Encoder model
            
            encoder =  Model(inputs= [encoder_inputs],outputs = [z_mean,z_variance,latent_vector],name='Encoder_model')
            
            #############################################################
            ########################## ATTENTION ########################
            ############################################################
            #https://www.joao-pereira.pt/publications/accepted_version_ICMLA18.pdf
            # Self attention layer, input encoded data saq. output seq of context vector
            #attention_inputs = Input(shape=[seq_length,features],name='att_input')
            attention_layer=SeqSelfAttention(attention_activation='sigmoid')(z1) #The sigmoid will allow you to have high probability for all of your classes, some of them, or none of them.
            
            attention_z_mean = Dense(coding_size,kernel_initializer=initializer,name='attention_Z_mean')(attention_layer)
            attention_z_variance = Dense(coding_size,kernel_initializer=initializer,name='attention_Z_variance')(attention_layer)
            
            attention_z_mean = Flatten()(attention_z_mean)
            attention_z_variance = Flatten()(attention_z_variance)
            
            attention_z_mean= Dense(coding_size,kernel_initializer=initializer)(attention_z_mean)
            attention_z_variance= Dense(coding_size,kernel_initializer=initializer,activation=activations.softplus)(attention_z_variance)
            
            
            attention_latent_vector = Sampling2()([attention_z_mean,attention_z_variance])
            
            
            #attention_x = Dense(seq_length*features,name='attention_dense')(attention_latent_vector)
            #attention_output = layers.Reshape([seq_length,features],name='attention_reshape')(attention_x)
            #attention_output = Flatten(name='attention_flatten')(attention_output)
            #attention_output = TimeDistributed(Dense(50))(attention_output)
            #attention_output = Dense(coding_size,name='attention_output')(attention_output)
            #attention_output = layers.Reshape([-1,50],name='decoder_output')(attention_output)
            
            attention = Model(inputs = [encoder_inputs],outputs=[attention_z_mean,attention_z_variance,attention_latent_vector],name='Attention_model')
            
            
            #attention model
            
            #############################################################
            ########################## DECODER ########################
            ############################################################
            #Decoder Layers. 
            #masking_decoder = Masking(mask_value=0,input_shape=(None,coding_size))(decoder_inputs)
            x = RepeatVector(seq_length)(decoder_inputs)
            x = Bidirectional(LSTM(blstm[1],
                                   recurrent_dropout=recurrent_dropout, 
                                   kernel_regularizer=l2(blstm_regu), 
                                   recurrent_regularizer=l2(blstm_regu), 
                                   bias_regularizer=l2(blstm_regu),
                                   name='decoder_blstm1',
                                   return_sequences= True))(x)
            
            x = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(x)
            x = Dropout(dropout)(x)
            
            x = Bidirectional(LSTM(blstm[0],
                                   recurrent_dropout=recurrent_dropout, 
                                   kernel_regularizer=l2(blstm_regu), 
                                   recurrent_regularizer=l2(blstm_regu), 
                                   bias_regularizer=l2(blstm_regu),
                                   name='decoder_blstm2',
                                   return_sequences= True))(x)
            
            x = TimeDistributed(Dense(features,name='decoder_dense'))(x)
            #x = Dense(features,name='decoder_dense')(x)
            x = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(x)
            x = Dropout(dropout)(x)
            
            #decoder_output = layers.Reshape([seq_length,features],name='decoder_output')(x)
            #Decoder model
            decoder = Model(inputs = [decoder_inputs],outputs=[x],name='Decoder_model')
            
            #getting latent vector from encoder to decoder(only thing it needs)...
            _,_, latent = encoder(encoder_inputs)
            #getting attention
            _,_, att_latent = attention(encoder_inputs)
            decoder_inputs_attention = Concatenate()([att_latent, latent])
            #making reconstruction
            reconstuctions = decoder(decoder_inputs_attention)
            
            #Variational Autoencoder Model - combined.
            vae_model = Model(inputs=[encoder_inputs],outputs = [reconstuctions],name=name)
            
            #############################################################
            ##########################   Loss   ########################
            ############################################################
            
            #KL Loss for the reconstuction loss
            log_likelohood_regul = log_likelohood_regul
            DL_regu= DL_regu
            latent_loss = log_likelohood_regul*(-0.5)*K.sum(1+z_variance-K.exp(z_variance)-K.square(z_mean),axis=1)+DL_regu* custom_loss_regularisation*tf.norm(latent,1)
            vae_model.add_loss(K.mean(latent_loss)/(seq_length*features))
            
            #defining optizer
            if optimizer=='Nadam':
                optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
            elif optimizer=='Adam':
                optimizer = keras.optimizers.Adam(lr=lr_init,amsgrad=True,clipnorm=1)
            else:
                optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
                
            
            vae_model.compile(loss=tf.keras.losses.Huber(),
                              optimizer=optimizer,
                              metrics=[tf.keras.metrics.KLDivergence(),
                                       tf.keras.metrics.CosineSimilarity(axis=1)])
        
        
    else:
        #inputs for encoder and decoder
        encoder_inputs = Input(shape=[seq_length,features],name='encoder_input')
        decoder_inputs = Input(shape=(coding_size*2),name='decoder_input')
        #masking layer for 0 values
        #masking = Masking(mask_value=0,input_shape=(None,features),name='encoder_masking')(encoder_inputs)
        #############################################################
        ########################## ENCODER ########################
        ############################################################
        #Encoder layers
        z1 = Bidirectional(LSTM(blstm[0],name='encoder_blstm1',return_sequences= True))(encoder_inputs)
        z = Bidirectional(LSTM(blstm[1],name='encoder_blstm2'))(z1)
        z_mean = Dense(coding_size,name='encoder_Z_mean')(z)
        z_variance = Dense(coding_size,name='encoder_Z_variance',activation=activations.softplus)(z)
        latent_vector = Sampling2()([z_mean,z_variance])
        #Encoder model
        
        encoder =  Model(inputs= [encoder_inputs],outputs = [z_mean,z_variance,latent_vector],name='Encoder_model')
        
        #############################################################
        ########################## ATTENTION ########################
        ############################################################
        #https://www.joao-pereira.pt/publications/accepted_version_ICMLA18.pdf
        # Self attention layer, input encoded data saq. output seq of context vector
             
        
        
        
        attention_layer=SeqSelfAttention(attention_activation='sigmoid')(z1)
        attention_z_mean = Dense(coding_size,name='attention_Z_mean')(attention_layer)
        attention_z_variance = Dense(coding_size,name='attention_Z_variance',activation=activations.softplus)(attention_layer)
        attention_latent_vector = Sampling2()([attention_z_mean,attention_z_variance])
        #attention_x = Dense(seq_length*features,name='attention_dense')(attention_latent_vector)
        #attention_output = layers.Reshape([seq_length,features],name='attention_reshape')(attention_x)
        attention_output = Flatten(name='attention_flatten')(attention_latent_vector)
        #attention_output = TimeDistributed(Dense(50))(attention_output)
        attention_output = Dense(coding_size,name='attention_output')(attention_output)
        #attention_output = layers.Reshape([-1,50],name='decoder_output')(attention_output)
        attention = Model(inputs = [encoder_inputs],outputs=[attention_output],name='Attention_model')
        
        
        #attention model
        
        #############################################################
        ########################## DECODER ########################
        ############################################################
        #Decoder Layers. 
        #masking_decoder = Masking(mask_value=0,input_shape=(None,coding_size))(decoder_inputs)
        x = RepeatVector(seq_length)(decoder_inputs)
        x = Bidirectional(LSTM(blstm[1],name='decoder_blstm1',return_sequences= True))(x)
        x = Bidirectional(LSTM(blstm[0],name='decoder_blstm2'))(x)
        x = Dense(seq_length*features,name='decoder_dense')(x)
        decoder_output = layers.Reshape([seq_length,features],name='decoder_output')(x)
        #Decoder model
        decoder = Model(inputs = [decoder_inputs],outputs=[decoder_output],name='Decoder_model')
        
        #getting latent vector from encoder to decoder(only thing it needs)...
        _,_, latent = encoder(masking)
        #getting attention
        att = attention(masking)
        decoder_inputs_attention = Concatenate()([att, latent])
        #making reconstruction
        reconstuctions = decoder(decoder_inputs_attention)
        
        #Variational Autoencoder Model - combined.
        vae_model = Model(inputs=[encoder_inputs],outputs = [reconstuctions],name=name)
        
        #############################################################
        ##########################   Loss   ########################
        ############################################################
        
        
        #KL Loss for the reconstuction loss
        latent_loss = -0.5*K.sum(1+z_variance-K.exp(z_variance)-K.square(z_mean),axis=1)+ custom_loss_regularisation*tf.norm(latent,1)
        vae_model.add_loss(K.mean(latent_loss)/(seq_length*features))
        
        #defining optizer
        if optimizer=='Nadam':
            optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
        if optimizer=='Adam':
            optimizer = keras.optimizers.Adam(lr=lr_init,clipnorm=1)
        else:
            optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
        
        vae_model.compile(loss=general_loss,optimizer=optimizer,metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.CosineSimilarity(axis=1)])
    
    #############################################################
    ##########################   MISC   ########################
    ############################################################    
    if verbose>0:
        print(f"Number of Parameters: {vae_model.count_params()}\n")
        vae_model.summary()
    
    if verbose>1:
        encoder.summary()
        decoder.summary()
        
    
    input_variable = [seq_length,features,coding_size,coding_size,with_gpu,blstm,lr_init,general_loss,optimizer]
    with open(f'{path}/input_parameters{name}.pkl', 'wb') as sa:
        pickle.dump(input_variable, sa)
        print(f"input varibles saved in: {sa}")
    
    
    
    return vae_model,encoder,decoder,attention
        
def vae(seq_length = 310,
           features = 240,
           coding_size = 50,
           with_gpu=False,
           blstm = [200,100],
           lr_init =0.001,
           general_loss = 'cosine_similarity',
           optimizer  = 'Nadam',
           verbose=0
          ):
    '''
    
    Function to make a Variational Autoencoder.
    -------------------------------------------
    This function makes a Variational Autoencoder with Bidirectional Long Short Term Memory. This VAE can be used to:
    1) Detect anomalies in data (data similar to the type used for training)
    2) Create synthetic data similar to the data used for training.
    
    The VAE is inherently probibalistic.Here, a Gaussian VAE is made. For mode information on VAE, see the thesis.
    
    Firstly, an ENCODER is made. Then, the latent vector is computed (also called bottleneck). 
    Then, the DECODER is made. This decoder takes the latent vector and tries to compute the original sequence. 
    
    This is all combines in the model call vae.
    
    The input parameters are all saved in a folder designated for this model. These paramters should thus be used to create the model at a later data.
    The vae weights and history will similary be saved in this folder. 
    The history can be accesed as a dataframe and consists of the loss and acc.
    The weights can be loaded on the model at a later date
    
    Since this model uses costum layers, e.g. the Sampling() layer, the model cant be saved as a Keras model.
    Instead, the weights are saved. Consequently, to recontruct the model at a later date do the following:
        1) Re-create the model using the input parameters in the model folder.
        2) load the weigths from the model folder on the newly created model.
        3) Use the model with the corect weights for whatever you want.
        
    --------------------------------------------
    Input:
           seq_length = samples_scaled_padded_reshaped.shape[1],
           features = samples_scaled_padded_reshaped.shape[2],
           coding_size = 50,
           with_gpu=False,
           blstm = [200,100],
           lr_init =0.001,
           general_loss = 'cosine_similarity',
           optimizer  = 'Nadam',
           verbose=0
    Output:
           vae_model: The variational autoencoder
           encoder: The Encoder
           decode: The Decoder
           
    Example: 
           vae_model,encoder,decoder= cl_vae()
    
    Author:
           Kristian Soerensen
           kristian.sorensen@hotmail.com
           November 2020.
    Note:           
    '''
    print("Warning: Only the weights and history can be saved.Due to the costum layer 'Sampling', model can no be saved. Therefore, when loading the model, save the inputs to create the omdel anew and then load weight ontop of it.\n")
    
    print("Warning: Only the weights and history can be saved.Due to the costum layer 'Sampling', model can no be saved. Therefore, when loading the model, save the inputs to create the omdel anew and then load weight ontop of it.\n")
    
    name = f"VAE_model_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    path = name
    os.makedirs(path, exist_ok=True) 
    custom_loss_regularisation=0.000000001
    if with_gpu==True:
        if with_gpu==True and gpu_single==False:
            #strategy = tf.distribute.MirroredStrategy()
            strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy().scope()
        if with_gpu==True and gpu_single==True:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            #tf.config.experimental.set_memory_growth(gpus, True)
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(gpus[int(gpu_single_number)], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=29000)])
                except RuntimeError as e:
                    print(e)
                    
            strategy = tf.device(f"/GPU:{gpu_single_number}")
            
        if verbose>0:
            print('Number of devices: \n{}'.format(strategy.num_replicas_in_sync))
            
            
        with strategy:
            #inputs for encoder and decoder
            encoder_inputs = Input(shape=[seq_length,features],name='encoder_input')
            decoder_inputs = Input(shape=(coding_size),name='decoder_input')
            #masking layer for 0 values
            masking = Masking(mask_value=0,input_shape=(None,features),name='encoder_masking')(encoder_inputs)
            #############################################################
            ########################## ENCODER ########################
            ############################################################
            #Encoder layers
            z1 = Bidirectional(LSTM(blstm[0],
                                    recurrent_dropout=0.25, 
                                    kernel_regularizer=l2(0.0000001), 
                                    recurrent_regularizer=l2(0.0000001), 
                                    bias_regularizer=l2(0.0000001),
                                    name='encoder_blstm1',
                                    return_sequences= True))(masking)
            #file:///C:/Users/krist/AppData/Local/Temp/1602.02282.pdf
            z = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(z1)
            z = Dropout(0.5)(z)
            z = Bidirectional(LSTM(blstm[1],
                                   recurrent_dropout=0.25, 
                                   kernel_regularizer=l2(0.0000001), 
                                   recurrent_regularizer=l2(0.0000001), 
                                   bias_regularizer=l2(0.0000001),
                                   name='encoder_blstm2'))(z)
            z = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(z)
            z = Dropout(0.5)(z)
            
            z_mean = Dense(coding_size,name='encoder_Z_mean')(z)
            z_variance = Dense(coding_size,name='encoder_Z_variance',activation=activations.softplus)(z)
            
            latent_vector = Sampling2()([z_mean,z_variance])
            #Encoder model
            
            encoder =  Model(inputs= [encoder_inputs],outputs = [z_mean,z_variance,latent_vector],name='Encoder_model')
            
            
            #############################################################
            ########################## DECODER ########################
            ############################################################
            #Decoder Layers. 
            masking_decoder = Masking(mask_value=0,input_shape=(None,coding_size))(decoder_inputs)
            x = RepeatVector(seq_length)(masking_decoder)
            x = Bidirectional(LSTM(blstm[1],
                                   recurrent_dropout=0.25, 
                                   kernel_regularizer=l2(0.0000001), 
                                   recurrent_regularizer=l2(0.0000001), 
                                   bias_regularizer=l2(0.0000001),
                                   name='decoder_blstm1',
                                   return_sequences= True))(x)
            x = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(x)
            x = Dropout(0.5)(x)
            x = Bidirectional(LSTM(blstm[0],
                                   recurrent_dropout=0.25, 
                                   kernel_regularizer=l2(0.0000001), 
                                   recurrent_regularizer=l2(0.0000001), 
                                   bias_regularizer=l2(0.0000001),
                                   name='decoder_blstm2',
                                   return_sequences= True))(x)
            x = TimeDistributed(Dense(features,name='decoder_dense'))(x)
            x = BatchNormalization(momentum=0.99, 
                                   scale=False, 
                                   center=False,
                                   trainable=False)(x)
            x = Dropout(0.5)(x)
            
            
            decoder = Model(inputs = [decoder_inputs],outputs=[x],name='Decoder_model')
            
            #getting latent vector from encoder to decoder(only thing it needs)...
            _,_, latent = encoder(masking)
            #making reconstruction
            reconstuctions = decoder(latent)
            
            #Variational Autoencoder Model - combined.
            vae_model = Model(inputs=[encoder_inputs],outputs = [reconstuctions],name=name)
            
            #############################################################
            ##########################   Loss   ########################
            ############################################################
            
            #KL Loss for the reconstuction loss
            log_likelohood_regul = 0.001
            DL_regu= 0.001
            latent_loss = log_likelohood_regul*(-0.5)*K.sum(1+z_variance-K.exp(z_variance)-K.square(z_mean),axis=1)+DL_regu* custom_loss_regularisation*tf.norm(latent,1)
            vae_model.add_loss(K.mean(latent_loss)/(seq_length*features))
            
            #defining optizer
            if optimizer=='Nadam':
                optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
            elif optimizer=='Adam':
                optimizer = keras.optimizers.Adam(lr=lr_init,amsgrad=True,clipnorm=1)
            else:
                optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
                
            
            vae_model.compile(loss='cosine_similarity',optimizer=optimizer,metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.CosineSimilarity(axis=1)])
        
        
    else:
        #inputs for encoder and decoder
        encoder_inputs = Input(shape=[seq_length,features],name='encoder_input')
        decoder_inputs = Input(shape=(coding_size*2),name='decoder_input')
        #masking layer for 0 values
        masking = Masking(mask_value=0,input_shape=(None,features),name='encoder_masking')(encoder_inputs)
        #############################################################
        ########################## ENCODER ########################
        ############################################################
        #Encoder layers
        z1 = Bidirectional(LSTM(blstm[0],name='encoder_blstm1',return_sequences= True))(masking)
        z = Bidirectional(LSTM(blstm[1],name='encoder_blstm2'))(z1)
        z_mean = Dense(coding_size,name='encoder_Z_mean')(z)
        z_variance = Dense(coding_size,name='encoder_Z_variance',activation=activations.softplus)(z)
        latent_vector = Sampling2()([z_mean,z_variance])
        #Encoder model
        
        encoder =  Model(inputs= [encoder_inputs],outputs = [z_mean,z_variance,latent_vector],name='Encoder_model')
        
        #############################################################
        ########################## ATTENTION ########################
        ############################################################
        #https://www.joao-pereira.pt/publications/accepted_version_ICMLA18.pdf
        # Self attention layer, input encoded data saq. output seq of context vector
             
        
        
        
        attention_layer=SeqSelfAttention(attention_activation='sigmoid')(z1)
        attention_z_mean = Dense(coding_size,name='attention_Z_mean')(attention_layer)
        attention_z_variance = Dense(coding_size,name='attention_Z_variance',activation=activations.softplus)(attention_layer)
        attention_latent_vector = Sampling2()([attention_z_mean,attention_z_variance])
        #attention_x = Dense(seq_length*features,name='attention_dense')(attention_latent_vector)
        #attention_output = layers.Reshape([seq_length,features],name='attention_reshape')(attention_x)
        attention_output = Flatten(name='attention_flatten')(attention_latent_vector)
        #attention_output = TimeDistributed(Dense(50))(attention_output)
        attention_output = Dense(coding_size,name='attention_output')(attention_output)
        #attention_output = layers.Reshape([-1,50],name='decoder_output')(attention_output)
        attention = Model(inputs = [encoder_inputs],outputs=[attention_output],name='Attention_model')
        
        
        #attention model
        
        #############################################################
        ########################## DECODER ########################
        ############################################################
        #Decoder Layers. 
        masking_decoder = Masking(mask_value=0,input_shape=(None,coding_size))(decoder_inputs)
        x = RepeatVector(seq_length)(masking_decoder)
        x = Bidirectional(LSTM(blstm[1],name='decoder_blstm1',return_sequences= True))(x)
        x = Bidirectional(LSTM(blstm[0],name='decoder_blstm2'))(x)
        x = Dense(seq_length*features,name='decoder_dense')(x)
        decoder_output = layers.Reshape([seq_length,features],name='decoder_output')(x)
        #Decoder model
        decoder = Model(inputs = [decoder_inputs],outputs=[decoder_output],name='Decoder_model')
        
        #getting latent vector from encoder to decoder(only thing it needs)...
        _,_, latent = encoder(masking)
        #getting attention
        att = attention(masking)
        decoder_inputs_attention = Concatenate()([att, latent])
        #making reconstruction
        reconstuctions = decoder(decoder_inputs_attention)
        
        #Variational Autoencoder Model - combined.
        vae_model = Model(inputs=[encoder_inputs],outputs = [reconstuctions],name=name)
        
        #############################################################
        ##########################   Loss   ########################
        ############################################################
        
        
        #KL Loss for the reconstuction loss
        latent_loss = -0.5*K.sum(1+z_variance-K.exp(z_variance)-K.square(z_mean),axis=1)+ custom_loss_regularisation*tf.norm(latent,1)
        vae_model.add_loss(K.mean(latent_loss)/(seq_length*features))
        
        #defining optizer
        if optimizer=='Nadam':
            optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
        if optimizer=='Adam':
            optimizer = keras.optimizers.Adam(lr=lr_init,clipnorm=1)
        else:
            optimizer = keras.optimizers.Nadam(lr=lr_init,clipnorm=1)
        
        vae_model.compile(loss='cosine_similarity',optimizer=optimizer,metrics=[tf.keras.metrics.RootMeanSquaredError(),tf.keras.metrics.CosineSimilarity(axis=1)])
    
    #############################################################
    ##########################   MISC   ########################
    ############################################################    
    if verbose>0:
        print(f"Number of Parameters: {vae_model.count_params()}\n")
        vae_model.summary()
    
    if verbose>1:
        encoder.summary()
        decoder.summary()
        
    
    input_variable = [seq_length,features,coding_size,coding_size,with_gpu,blstm,lr_init,general_loss,optimizer]
    with open(f'{path}/input_parameters{name}.pkl', 'wb') as sa:
        pickle.dump(input_variable, sa)
        print(f"input varibles saved in: {sa}")
    
    
    
    return vae_model,encoder,decoder,attention      
    
    
class CustomModelCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # logs is a dictionary
            
            if epoch%10==0: 
                self.model.save(f'{self.model.name}/model_{self.model.name}.tf', overwrite=True)
                self.model.save_weights(f'{self.model.name}/weights_{self.model.name}.tf', overwrite=True)
                pd.DataFrame(self.model.history.history).to_pickle(f"{self.model.name}/history_epoch_{epoch}_{self.model.name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}.pkl")
                

def get_callbacks(model):
    '''
    '''
    os.makedirs(f"{model.name}", exist_ok=True) 
    name = f"{model.name}_par_{model.count_params()}"
    best_model_file = f"{model.name}/best_model_{name}.tf"
    #best_model_file_manual_save = f"model/manual_save{name}.h5"
    os.makedirs(f"{model.name}/logs", exist_ok=True) 
    logdir = f"{model.name}/logs/log_{name}"



    early_stop = EarlyStopping(monitor='loss', patience=15)
    
    ################ CALLBACKS ####################
    best_model = ModelCheckpoint(best_model_file, 
                                     monitor='loss', 
                                     mode='auto',
                                     verbose=1, 
                                     save_best_only=True)
        
    #reduce lr when learning slows down...
    reduce_lr = ReduceLROnPlateau(monitor='loss',
                                      factor=0.1,
                                      patience=5, 
                                      min_lr=0.00000000000000000001)
    
    
    
                
                
    cbk = CustomModelCheckpoint()    
    callbacks=[reduce_lr,best_model,early_stop,cbk]
    return callbacks