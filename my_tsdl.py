import datetime
from keras.engine import training
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,LSTM, Dropout, Bidirectional, SimpleRNN, GRU, TimeDistributed, Input, RNN,RepeatVector,Masking,TimeDistributed,BatchNormalization
from tensorflow.keras.experimental import PeepholeLSTMCell
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
import datetime, os
import tensorflow as tf
import keras
import pandas as pd #to create df for resutls.

import matplotlib.pyplot as plt # for plotting

'''
###############################################################
########### time series deep learning(tdsl) ###################
###############################################################
This bla bla bla

LSTM
Bidirectional LSTM
Sequence 2 Sequence Encoder Decoder LSTM


'''    

def seq2seq(input_shape_sample=[95,2],
            input_shape_target=[2,1],
            Name='seq2seq',
            n_hidden_coder=100,
            n_hidden_blstm=[200,100],
            recurrent_dropout=[0.2,0.2],
            kernel_regularizerl2=[0.000001,0.000001],
            recurrent_regularizerl2=[0.000001,0.000001],
            bias_regularizer=[0.000001,0.000001],
            BatchNormalization_encoder = [0.6], #momentum currently.
            BatchNormalization_decoder = [0.6],
            encoder_dropout=0.02,
            encoder_recurrentdrop=0.002,
            decoder_dropout=0.02,
            decoder_recurrentdrop=0.002,
            n_hidden_blstm_decoder = [50],
            verbose = 0) -> training.Model:
    '''
    nice
    example: 
        seq2seq(input_shape_x=x_train_scaled_padded[0],
                input_shape_target= y_train_scaled_padded[0]
                Name,
                
    
    '''
    #####input
    input_train = Input(shape=(input_shape_sample[0], input_shape_sample[1]))
    output_train = Input(shape=(input_shape_target[0], input_shape_target[1]))

    #Masking
    x = Masking(mask_value=-9999,
                input_shape=(None,
                             input_shape_sample[1],
                             input_shape_sample[0]))(input_train)
    
    ############## BLSTM BEFORE ENCODER ####################
    counter_encoder = 0
    for i in n_hidden_blstm:
        x = Bidirectional(LSTM(n_hidden_blstm[counter_encoder],
                               recurrent_dropout=recurrent_dropout[counter_encoder], 
                               kernel_regularizer=l2(kernel_regularizerl2[counter_encoder]), 
                               recurrent_regularizer=l2(recurrent_regularizerl2[counter_encoder]), 
                               bias_regularizer=l2(bias_regularizer[counter_encoder]), 
                               return_sequences= True,
                               name=f"encoder_BLSTM_{counter_encoder+1}"))(x)
        counter_encoder = counter_encoder+1



    ############### ENCODER ###############################
    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(n_hidden_coder,
                                                            dropout=encoder_dropout, 
                                                            recurrent_dropout=encoder_recurrentdrop,
                                                            return_sequences=False,
                                                            return_state=True,
                                                            name = 'encoder_layer')(x)
    ################# BATCH NORMALIZATION OF ENCODER ###############3
    encoder_last_h1 = BatchNormalization(momentum=BatchNormalization_encoder[0],name = 'encoder_hidden_normalization')(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=BatchNormalization_encoder[0],name = 'encoder_output_normalization')(encoder_last_c)


    #################### DECODER #######################
    #Next, we make 20 copies of the last hidden state of encoder and use them as input to the decoder. 
    #The last cell state and the last hidden state of the encoder are also used as the initial states of decoder.
    decoder = RepeatVector(output_train.shape[1],name = 'decoder_input')(encoder_last_h1)
    decoder = LSTM(n_hidden_coder,
                   dropout=decoder_dropout, 
                   recurrent_dropout=decoder_recurrentdrop, 
                   return_state=False, 
                   return_sequences=True,
                   name = 'decoder_layer')(decoder, initial_state=[encoder_last_h1, encoder_last_c])
    counter_decoder = 0
    
    #################### BLSTM AFTER DECODER ##############
    for j in n_hidden_blstm_decoder:       
        decoder = Bidirectional(LSTM(n_hidden_blstm_decoder[counter_decoder],
                               recurrent_dropout=0.2, 
                               kernel_regularizer=l2(0.000001), 
                               recurrent_regularizer=l2(0.000001), 
                               bias_regularizer=l2(0.000001), 
                               return_sequences= True, name=f"decoder_BLSTM_{counter_decoder+1}"))(decoder)
        counter_decoder = counter_decoder+1
        
        
    
    out = TimeDistributed(Dense(1))(decoder)
    #out = Dense(1)(decoder)
    

    
    if Name!='seq2seq':
        Name=Name
    else:
        Name = Name+f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    model = Model(inputs=input_train, outputs=out,name=Name)
    if verbose>0:
        model.summary()
    
    return model
    
    
def blmst(input_shape_sample_blstm=[2,30],
          Name='blstm',
          BatchNormalization_blstm=[0.99,0.0001],
          bltsm_dropout=[0.02,0.02],
          n_hidden_blstm=[200,100],
          recurrent_dropout=[0.2,0.2],
          kernel_regularizerl2=[0.000001,0.000001],
          recurrent_regularizerl2=[0.000001,0.000001],
          bias_regularizer=[0.000001,0.000001],
          verbose = 0) -> training.Model:
    '''
    '''
    input_train = Input(shape=(input_shape_sample_blstm[0], input_shape_sample_blstm[1]))
    
    #Masking
    x = Masking(mask_value=0,
                input_shape=(None,
                             input_shape_sample_blstm[0],
                             input_shape_sample_blstm[1]))(input_train)
    counter_encoder = 0
    for i in n_hidden_blstm:
        x = Bidirectional(LSTM(n_hidden_blstm[counter_encoder],
                               recurrent_dropout=recurrent_dropout[counter_encoder], 
                               kernel_regularizer=l2(kernel_regularizerl2[counter_encoder]), 
                               recurrent_regularizer=l2(recurrent_regularizerl2[counter_encoder]), 
                               bias_regularizer=l2(bias_regularizer[counter_encoder]), 
                               return_sequences= True,
                               name=f"encoder_BLSTM_{counter_encoder+1}"))(x)
        x = BatchNormalization(momentum=BatchNormalization_blstm[0],epsilon=BatchNormalization_blstm[1],name = f"encoder_output_normalization_{counter_encoder+1}")(x)
        x = Dropout(bltsm_dropout[counter_encoder])(x)
        counter_encoder = counter_encoder+1     
        
    out =TimeDistributed(Dense(1))(x)
    
    if Name!='blstm':
        Name=Name
    else:
        Name = Name+f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    model = Model(inputs=input_train, outputs=out,name=Name)
    if verbose>0:
        model.summary()
    
    return model



def blmst_2(input_shape_sample_blstm=[2,30],
          Name='blstm',
          BatchNormalization_blstm=[0.99,0.0001],
          bltsm_dropout=[0.02,0.02],
          n_hidden_blstm=[200,100],
          recurrent_dropout=[0.2,0.2],
          kernel_regularizerl2=[0.000001,0.000001],
          recurrent_regularizerl2=[0.000001,0.000001],
          bias_regularizer=[0.000001,0.000001],
          verbose = 0) -> training.Model:
    '''
    '''
    input_train = Input(shape=(input_shape_sample_blstm[0], input_shape_sample_blstm[1]))
    
    #Masking
    x = Masking(mask_value=-9999,
                input_shape=(None,
                             input_shape_sample_blstm[0],
                             input_shape_sample_blstm[1]))(input_train)
    counter_blstm = 0
    for i in n_hidden_blstm:
        x = Bidirectional(LSTM(n_hidden_blstm[counter_blstm],
                               recurrent_dropout=recurrent_dropout[counter_blstm], 
                               kernel_regularizer=l2(kernel_regularizerl2[counter_blstm]), 
                               recurrent_regularizer=l2(recurrent_regularizerl2[counter_blstm]), 
                               bias_regularizer=l2(bias_regularizer[counter_blstm]), 
                               return_sequences= True,
                               name=f"BLSTM_{counter_blstm+1}"))(x)
        x = BatchNormalization(momentum=BatchNormalization_blstm[0],epsilon=BatchNormalization_blstm[1],name = f"Normalization_{counter_blstm+1}")(x)
        x = Dropout(bltsm_dropout[counter_blstm])(x)
        counter_blstm = counter_blstm+1     
        
    out =TimeDistributed(Dense(1))(x)
    
    if Name!='blstm':
        Name=Name
    else:
        Name = Name+f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    model = Model(inputs=input_train, outputs=out,name=Name)
    if verbose>0:
        model.summary()
    
    return model

def lstm(input_shape_sample,n_hidden_lstm=[50],Name='lstm',verbose=0):
    '''
    
    '''
    input_train = Input(shape=(input_shape_sample[0], 
                               input_shape_sample[1]))
    
    x = Masking(mask_value=-9999,
                input_shape=(None,
                             input_shape_sample[0],
                             input_shape_sample[1]))(input_train)
    counter = 0
    for i in n_hidden_lstm:
        x = (LSTM(n_hidden_lstm[counter],
                  return_sequences= True,
                  name=f"LSTM_{counter+1}"))(x)
        counter = counter+1 
        
    out =(Dense(1))(x)
    
    if Name!='lstm':
        Name=Name
    else:
        Name = Name+f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    model = Model(inputs=input_train, outputs=out,name=Name)
    if verbose>0:
        model.summary()
    


    return model
    
    
    
def train_model(model,x_train,y_train,x_val,y_val,batchsize=1,epoch=1,lr_init=0.1,verbose=0,name='model'):
    '''
    
    '''
    keras.backend.clear_session()
    
    best_model_file = f"{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h5"
    logdir = f"logs/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    ################ CALLBACKS ####################
    best_model = ModelCheckpoint(best_model_file, 
                                 monitor='val_loss', 
                                 mode='min',verbose=verbose-1, 
                                 save_best_only=True)
    
    #reduce lr when learning slows down...
    reduce_lr = ReduceLROnPlateau(monitor='mse',
                                  factor=0.1,
                                  patience=10, 
                                  min_lr=0.0000001)
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                histogram_freq=1)
    
    callbacks=[reduce_lr]
    
    optimizer = Adam(lr=lr_init)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    
    mse = []
    val_mse = []
    loss = []
    val_loss = []
    for ep_sa in range(epoch):
        for nb_epo in range(x_val.shape[0]):
            for i in range(len(x_train)):
                model.reset_states()
                history= model.fit(x_train[i],
                                        y_train[i][:,:,None],
                                        epochs=epoch,
                                        batch_size=batchsize,
                                        verbose=verbose-1,
                                        shuffle=False,
                                        validation_data=(x_val[nb_epo], y_val[nb_epo][:,:,None]),
                                        callbacks=callbacks)
            
            
        
                mse.extend(history.history['mse'])
                val_mse.extend(history.history['val_mse'])
                loss.extend(history.history['loss'])
                val_loss.extend(history.history['val_loss'])
        print(f"Training seq {i+1} of {len(x_train)+1}. Loss: {history.history['loss'][-1]}") 
                 

    model.save(best_model_file)
    d = {'MSE': mse, 'Validation MSE': val_mse, 'Loss': loss, 'Validation loss': val_loss}

    df = pd.DataFrame(data=d)

    return model, df, history


def df2plot(df):
    '''
     Df with mse, mse val, loss, loss val.
     
    '''
    # Plot history: MSE
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    #ax1.semilogy(np.array(acc_vali))
    ax1.semilogy(df.MSE,color='tab:blue')
    #plt.title('MSE for Chennai Reservoir Levels')
    plt.ylabel('MSE value')
    plt.xlabel('No. sequence')
    plt.title('MSE value.')
    ax1.set_ylabel('MSE value',color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color)
    #plt.legend(loc="upper left")
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.semilogy(df['Validation MSE'],color='orange')
    ax2.set_ylabel('validation MSE', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.show()
    
    return None
    
    
def blmst_reshaped(
          samples_scaled_padded,
          Name='blstm',
          BatchNormalization_blstm=[0.99,0.0001],
          bltsm_dropout=[0.02,0.02],
          n_hidden_blstm=[200,100],
          recurrent_dropout=[0.2,0.2],
          kernel_regularizerl2=[0.000001,0.000001],
          recurrent_regularizerl2=[0.000001,0.000001],
          bias_regularizer=[0.000001,0.000001],
          verbose = 0):
    '''
    --------------------------------------------------------------
    A BLSTM model for reshaped data. 
    The model is "simple" in the sense that it only uses BLSTM layers as well as Masking, dense and dropout layers.
    
    This model takes in reshped input data of format [trajectories x sequences x features]  where features is a reshaped version of the entire window of features for many time steps.
    Features = [timestepsx ais_features] where ais_features is e.g. lat, lon.
    
    This model is used for a classification scheme with 51 classes (49+1+1).
    
    For more information on this model, see the thsis "..... navnet på min thesis"
    --------------------------------------------------------------
    Input:
        samples_scaled_padded[trajectory x batch x features]: numpy array of input data
        Name[string:'blstm']: Name of model. 
        BatchNormalization_blstm[list]: list of batch normalization values. 
        bltsm_dropout[list]: Dropout values for the blstm layer
        n_hidden_blstm[list]: list of int of neurons. The length of the list decided the amount of individual layers.
        kernel_regularizerl2[lsit]: list of values for l2 regularisation
        recurrent_regularizerl2[list]: same
        bias_regularizer[list]: same
        verbose[0]: Amount of info printed.
    --------------------------------------------------------------   
    output:
        model: A deep LSTM model for data of shape [trajectory x batch x features]
    --------------------------------------------------------------    
    Example:
    The following example makes a model with 1 BLSTM layer with 200 neurons. Nothing is printed.
        model_1layer = blmst_reshaped(
          samples_scaled_padded,
          Name='blstm',
          BatchNormalization_blstm=[0.99,0.0001],
          bltsm_dropout=[0.02],
          n_hidden_blstm=[200],
          recurrent_dropout=[0.2],
          kernel_regularizerl2=[0.000001],
          recurrent_regularizerl2=[0.000001],
          bias_regularizer=[0.000001],
          verbose = 0)
    The following examples makes a model with 2 BLSTM layers with 200 and 100 neurons respecticly. The model is shown.
        model_1layer = blmst_reshaped(
          samples_scaled_padded,
          Name='blstm',
          BatchNormalization_blstm=[0.99,0.0001],
          bltsm_dropout=[0.02,0.02],
          n_hidden_blstm=[200,100],
          recurrent_dropout=[0.2,0.2],
          kernel_regularizerl2=[0.000001,0.000001],
          recurrent_regularizerl2=[0.000001,0.000001],
          bias_regularizer=[0.000001,0.000001],
          verbose = 1)
    --------------------------------------------------------------      
    Asserts:
        - Wrong input lenghts.
    --------------------------------------------------------------    
    Author:
        Kristian Soerensen
        kristian.sorensen@hotmail.com
        November 2020
    
    '''
    assert len(bltsm_dropout)>0,print("No layers defined")
    assert (len(bltsm_dropout)==len(n_hidden_blstm)),print(f"Length of list should be the same. Length are: {len(bltsm_dropout)} and {len(n_hidden_blstm)}")
    assert len(n_hidden_blstm)==len(recurrent_dropout),print(f"Length of list should be the same. Length are: {len(n_hidden_blstm)} and {len(recurrent_dropout)}")
    assert len(recurrent_dropout)==len(kernel_regularizerl2),print(f"Length of list should be the same. Length are: {len(recurrent_dropout)} and {len(recurrent_regularizerl2)}") 
    assert len(recurrent_regularizerl2)==len(bias_regularizer),print(f"Length of list should be the same. Length are: {len(recurrent_regularizerl2)} and {len(bias_regularizer)}")
    assert len(samples_scaled_padded.shape)==3,print(f"Shape of input date is {samples_scaled_padded.shape}. Should be of shape (sequence,batch,features)")
    
    
    input_train = Input(batch_shape= (None, samples_scaled_padded.shape[1],samples_scaled_padded.shape[2]))

    x = Masking(mask_value=0,input_shape=(None,samples_scaled_padded.shape[2]))(input_train)
    
    
    counter_blstm = 0
    for i in n_hidden_blstm:
        x = Bidirectional(LSTM(n_hidden_blstm[counter_blstm],
                               recurrent_dropout=recurrent_dropout[counter_blstm], 
                               kernel_regularizer=l2(kernel_regularizerl2[counter_blstm]), 
                               recurrent_regularizer=l2(recurrent_regularizerl2[counter_blstm]), 
                               bias_regularizer=l2(bias_regularizer[counter_blstm]), 
                               return_sequences= True,
                               name=f"BLSTM_{counter_blstm+1}"))(x)
        x = BatchNormalization(momentum=BatchNormalization_blstm[0],epsilon=BatchNormalization_blstm[1],name = f"Normalization_{counter_blstm+1}")(x)
        x = Dropout(bltsm_dropout[counter_blstm])(x)
        counter_blstm = counter_blstm+1     
        
    out =TimeDistributed(Dense(51))(x)
    out = Dropout(bltsm_dropout[0])(out)
    out =(Dense(50+1,activation='softmax'))(out)
    
    if Name!='blstm':
        Name=Name
    else:
        Name = Name+f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    model = Model(inputs=input_train, outputs=out,name=Name)
    if verbose>0:
        model.summary()
    
    return model