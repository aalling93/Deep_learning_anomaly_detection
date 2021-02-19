
import sys
import numpy as np
import pandas as pd
import math
import os
#general 
import sys
import numpy as np
import pandas as pd
import math
import os
# RNN, deep leraning ect.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,GlobalMaxPool1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, SimpleRNN, GRU, TimeDistributed, ConvLSTM2D, RNN,Conv1D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.experimental import PeepholeLSTMCell

from tensorflow.keras.callbacks import TensorBoard


from time import time
from keras import backend as K

#plotting
#import gdal
#import osr
#import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



import math


from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

def spherical_distance(lat1, long1, lat2, long2):
    phi1 = 0.5*math.pi - lat1
    phi2 = 0.5*math.pi - lat2
    r = 0.5*(6378137 + 6356752) # mean radius in meters
    t = math.sin(phi1)*math.sin(phi2)*math.cos(long1-long2) + math.cos(phi1)*math.cos(phi2)
    return r * math.acos(t)



def ellipsoidal_distance(lat1, long1, lat2, long2):

    a = 6378137.0 # equatorial radius in meters 
    f = 1/298.257223563 # ellipsoid flattening 
    b = (1 - f)*a 
    tolerance = 1e-11 # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = math.atan((1-f)*math.tan(phi1))
    U2 = math.atan((1-f)*math.tan(phi2))
    L1, L2 = long1, long2
    L = L2 - L1

    lambda_old = L + 0

    while True:
    
        t = (math.cos(U2)*math.sin(lambda_old))**2
        t += (math.cos(U1)*math.sin(U2) - math.sin(U1)*math.cos(U2)*math.cos(lambda_old))**2
        sin_sigma = t**0.5
        cos_sigma = math.sin(U1)*math.sin(U2) + math.cos(U1)*math.cos(U2)*math.cos(lambda_old)
        sigma = math.atan2(sin_sigma, cos_sigma) 
    
        sin_alpha = math.cos(U1)*math.cos(U2)*math.sin(lambda_old) / (sin_sigma+0.00000001)
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*math.sin(U1)*(math.sin(U2)+0.00000001)/(cos_sq_alpha+0.00000001)
        C = f*cos_sq_alpha*(4 + f*(4-3*cos_sq_alpha))/16
    
        t = sigma + C*sin_sigma*(cos_2sigma_m + C*cos_sigma*(-1 + 2*cos_2sigma_m**2))
        lambda_new = L + (1 - C)*f*sin_alpha*t
        if abs(lambda_new - lambda_old) <= tolerance:
            break
        else:
            lambda_old = lambda_new

    u2 = cos_sq_alpha*((a**2 - b**2)/b**2)
    A = 1 + (u2/16384)*(4096 + u2*(-768+u2*(320 - 175*u2)))
    B = (u2/1024)*(256 + u2*(-128 + u2*(74 - 47*u2)))
    t = cos_2sigma_m + 0.25*B*(cos_sigma*(-1 + 2*cos_2sigma_m**2))
    t -= (B/6)*cos_2sigma_m*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigma_m**2)
    delta_sigma = B * sin_sigma * t
    s = b*A*(sigma - delta_sigma)

    return s

from sklearn.preprocessing import StandardScaler
def ais_scaler(X_train):
    '''
    '''
    scaler = StandardScaler()
    
    
    data_to_scale = X_train[0]
    for i in range(1,len(X_train),1):
        data_to_scale= np.append(data_to_scale,X_train[i],axis=0)
        
    scaler.fit(data_to_scale)
    
    return scaler

def dataset_grid(X_train_dist,lookback=40,targets_future=[1],resolution=0.7):
    '''
    col 1  = lon
    col 2 = lat
    '''
    X_train_samples =[]
    X_train_targets = []
    for i in range(len(X_train_dist)):
        X_train_samples_temp =[]
        X_train_targets_temp = []
        for steps in range(0,len(X_train_dist[i])-lookback-5,1):
            sample = X_train_dist[i][steps:steps+lookback,:]
            X_train_samples_temp.append(sample)
            targets = []
            for pred in range(len(targets_future)):
                targets_temp = X_train_dist[i][steps+lookback+targets_future[pred],0:2]
                targets_class = y_class(sample[-1,0:2]-targets_temp,resolution = resolution)
                targets.append(targets_class)
        
            X_train_targets_temp.append(targets)
        X_train_samples_temp = np.array(X_train_samples_temp)
        X_train_targets_temp = np.array(X_train_targets_temp)
        X_train_samples.append(X_train_samples_temp)
        X_train_targets.append(X_train_targets_temp)
    
    
    X_train_samples = np.array(X_train_samples)
    X_train_targets = np.array(X_train_targets) 
    
    return X_train_samples,X_train_targets

def add_dist(ais_dat):
    '''
    '''
    #change length 
    train_shorted_test = ais_dat.copy()
    training = []

    for k in range(len(train_shorted_test)):
        distance = [0] #cant stqart with 0
        for i in range((train_shorted_test[k].shape[0]-1)):
            distance.append(ellipsoidal_distance(train_shorted_test[k][i,0],train_shorted_test[k][i,1],train_shorted_test[k][i+1,0],train_shorted_test[k][i+1,1]) * 0.001)
        distance = np.array(distance)
        temp = np.c_[ train_shorted_test[k], distance ] 
        #temp = temp[1:,[0, 1,2,5]]
        temp2 = np.around(temp, decimals=4)
        training.append(np.array(temp))
        
    


    data_training = np.array(training) 

    return data_training



def add_dist_speed(ais_dat,sampling=5):
    '''
    '''
    #change length 
    train_shorted_test = ais_dat.copy()
    training = []

    for k in range(len(train_shorted_test)):
        distance = [0] #cant stqart with 0
        for i in range((train_shorted_test[k].shape[0]-1)):
            distance.append(ellipsoidal_distance(train_shorted_test[k][i,0],train_shorted_test[k][i,1],train_shorted_test[k][i+1,0],train_shorted_test[k][i+1,1]) * 0.001)
        distance = np.array(distance)
        speed = distance/(sampling*60)
        temp = np.c_[ train_shorted_test[k], distance,speed ] 
        #temp = temp[1:,[0, 1,2,5]]
        temp2 = np.around(temp, decimals=4)
        training.append(np.array(temp))
        
    


    data_training = np.array(training) 

    return data_training

def pad_data(data,max_lenght,value=0):
    '''
    '''
    return pad_sequences(data, padding='pre',value=value, dtype='float32', maxlen=max_lenght)

def max_lenght(data):
    '''
    '''
    max_lenght = 0
    for i in range(len(data)):
        if len(data[i])>max_lenght:
            max_lenght = len(data[i])
        
    return max_lenght

def create_dataset(trainX_scaled,split_validate_value=0.9,split_traintest=0.7,lookback=24,features=3,fatures_predict=3,verbose=0):
    '''
    
    Input
    
    The many sequences, each containing a timesereis of varying samples, are both the features and the target varible for the deep learning. 
    We are using the past coordinates to predict the future coordinates.
    
    This function is splitting the data into training and testing sets. 
    In timeseries, this can not be done randomly since the problem is sequential.
    We are therefore splitting it sequential.
    Thus, for each sequence: 
        The data is split into a training set and testing ste(defined by split_traintest)
    For all the sequences the data is split into a validatiaon set, (defined by split_validate_value), containting full sequences.
    
    For this model, the lookback parameter is important. The lookback 
    In order to predict a datapoint we ahve to provide an amount of datapoint, defined by the lookback.
    Meaning, we are not predicting anything for the first N points, defined by the lookback.
    For all other N+1, N+2... points, they will be predicted using  N points.
    
    The shape of the testing data is therefore defined from split_traintest and the lookback.
    
    For each sequence, The training set and testing set both will be reshaped into a 3 dimensional array for use with the LSTM.
    Thus, a 4 dimensional array is returned of shape (Number of sequence, number of samples,)
    
    For each sqeuence, we have X training sequences. Each of these training sequences hace a length defiend by lookback. 
    Each of these training sequences have a target.
    The number X is here defined from the original length of the sequence:
        X = Lenght_origianl_seq - Lookback - 1
        I.e. if the origianl length is 176 samples. Training at testing split is 0.7. The lookback is 24 samples, the size of the training set for this one seqeunce is:
        floor(176*0.7)-24-1 = 76.
        For this ONE sequence, there will therefore be 76 training sets, each with a length on 24. There is correspondingly 76 targets.
        
    '''
    assert features>1
    assert features<10
    
    
    if (trainX_scaled[0].shape[-1])>features:
        data = []
        for i in range(len(trainX_scaled)):
            data.append(trainX_scaled[i][:,0:features])
            
        data = np.array(data)
    else:
        data = trainX_scaled.copy()
        
    scaler = StandardScaler()
    
    X_validate =  data[math.floor(data.shape[0]*split_validate_value):]
    X_train_test = data[0:math.floor(data.shape[0]*split_validate_value)]
    
    X_train_full = X_train_test[0:math.floor(data.shape[0]*split_traintest)]
    X_test_full = X_train_test[math.floor(data.shape[0]*split_traintest):]
    
    if verbose>0:
        print('shape val ',X_validate.shape,'\nshape train test ',X_train_test.shape)
    
    train_size = math.floor(X_train_test.shape[0]*split_traintest)
    

    #X_train = []
    #X_test = []
    #the 
    #for i in range(X_train_test.shape[0]):
    #    X_train.append(X_train_test[i][0:math.floor(X_train_test[i].shape[0]*split_traintest),:])
    #    #the test set is getting additionally datasets, defined by Lookback, in toder to predict the very first value. 
    #    X_test.append(X_train_test[i][math.floor(X_train_test[i].shape[0]*split_traintest)-lookback:-1,:])
    
    
        
    X_train = np.array(X_train_full)
    X_test = np.array(X_test_full)
    
    if verbose>0:
        print('\nshape of train ',X_train.shape,'\nshape of test ',X_test.shape)
    #print('\nshape of first train ',X_train[0].shape,'\nshape of first test ',X_test[0].shape)
    
    data_to_scale = X_train[0]
    for i in range(1,len(X_train),1):
        data_to_scale= np.append(data_to_scale,X_train[i],axis=0)
        
    scaler.fit(data_to_scale)
    #print('scaling ',data_to_scale.shape)
    X_train_sample =[]
    X_train_target =[]
    
    X_test_sample =[]
    X_test_target =[]
    
    ############ Making testing set ################33
    if verbose>0:
        print('make test set')
    for i in range(X_test.shape[0]):
        if len(X_test[i])>lookback:
            #print('\nfor loop\n-----\n',X_test.shape[0])
            sample = []
            target =[]
            #print(X_test[i].shape[0])
            for j in range(X_test[i].shape[0]- lookback -1):   
                sample.append(np.array(X_test[i][j:(j+ lookback), :]))
                target.append(np.array(X_test[i][j + lookback, 0:fatures_predict]))
            
            sample = np.array(sample)
            target = np.array(target)
            
            sample = np.transpose(sample, (0, 2, 1))
            
            X_test_sample.append(sample)
            X_test_target.append(target)
        
        
    
    X_test_sample = np.array(X_test_sample)
    X_test_target = np.array(X_test_target)
    if verbose>0:
        print('make train set')
    #print('shape train before loop ',X_train.shape)
    #print('shape train before loop first sq ',X_train[0].shape)
    for i in range(X_train.shape[0]):
        if len(X_train[i])>lookback:
            sample = []
            target =[]
            #print(X_test[i].shape[0])
            for j in range(X_train[i].shape[0]- lookback -1):   
                sample.append(np.array(X_train[i][j:(j+ lookback), :]))
                target.append(np.array(X_train[i][j + lookback, 0:fatures_predict]))
                
            sample = np.array(sample)
            target = np.array(target)
            
            sample = np.transpose(sample, (0, 2, 1))

            X_train_sample.append(sample)
            X_train_target.append(target)
        
        
    
    X_train_sample = np.array(X_train_sample)
    X_train_target = np.array(X_train_target)
    
    
    validate_sample =[]
    validate_target =[]
    if verbose>0:
        print('make validation set')
    for i in range(X_validate.shape[0]):
        if len(X_validate[i])>lookback:
            #print('\nfor loop\n-----\n',X_test.shape[0])
            sample = []
            target =[]
            #print(X_test[i].shape[0])
            for j in range(X_validate[i].shape[0]- lookback -1):   
                sample.append(np.array(X_validate[i][j:(j+ lookback), :]))
                target.append(np.array(X_validate[i][j + lookback, 0:fatures_predict]))
            
            sample = np.array(sample)
            target = np.array(target)
            sample = np.transpose(sample, (0, 2, 1))

            validate_sample.append(sample)
            validate_target.append(target)
        
        
    
    validate_sample = np.array(validate_sample)
    validate_target = np.array(validate_target)
    
    
    
    return X_train_sample, X_train_target, X_test_sample, X_test_target, validate_sample, validate_target, scaler


import copy
import copy
def scale_data_outdated(data):
    '''
        To prepare the data for the Deep learning methods, it need to be scaled to a standard scale.
        We are here using the standard scaler in the sklearn library, in which the data is scaled to a mean value of 0.
        
        
    '''
    trainX_scaled = data.copy()
    scaler = StandardScaler()
    print("Lat Range before scaling for first sequence: " , 
          min(data[0][:,0]),
          max(data[0][:,0]))
    
    print("Lon Range before scaling for first sequence : " , 
          min(data[0][:,1]),
          max(data[0][:,1]))
    
    for i in range(trainX_scaled.shape[0]):
        trainX_scaled[i]=scaler.fit_transform(data[i][:,0:2])
        #If i dont want it saceled..
        #trainX_scaled[i]=data[i][:,0:2]
        
    print("lat Range after scaling : " , 
          min(trainX_scaled[0][:,0]),
          max(trainX_scaled[0][:,0]))

    print("lon Range after scaling : " , 
          min(trainX_scaled[0][:,1]),
          max(trainX_scaled[0][:,1]))
    
    return trainX_scaled,scaler

def scale_data(org_data,scaler):
    '''
        To prepare the data for the Deep learning methods, it need to be scaled to a standard scale.
        We are here using the standard scaler in the sklearn library, in which the data is scaled to a mean value of 0.
        
        
    '''
    #print(org_data.shape)
    data =copy.deepcopy(org_data)
    if (len(data[0].shape))==3:
        for i in range((data.shape[0])):
            for j in range(data[i].shape[-1]):               
                data[i][:,:,j]=scaler.transform(data[i][:,:,j])
    
    if (len(data[0].shape))==2:
        for i in range((data.shape[0])):
            data[i]=scaler.transform(data[i])

    return data

def prediction_targets(x,y,model,scal,seq=0):
    '''
    Getting the predicitons for x.
    
    Scaling the targets for y. (meaning for y, the data will just be inversly scaled and appended..)
    
    '''
    

    prediction1 = []
    true1 = []

    for j in range(x[seq].shape[0]):
        pred_temp = x[seq][j,:,:]
        pred_temp = model.predict(pred_temp[None,:,:])
        pred_temp =scaler.inverse_transform(pred_temp[:,:,0])
        true1.append(scaler.inverse_transform(y[seq][j]))
        prediction1.append(np.squeeze(pred_temp))
    
    true2 = np.array(true1)
    true2[true2==true2[0]]=np.nan

    prediction2 = np.array(prediction1)
    prediction2[prediction2==prediction2[0]]=np.nan
    
    return prediction2, true2

import copy



def plot_coord(lista):
    '''
    plotting coordinate on basemap;;
    
    lista : array of list of predictions...
    []
    
    '''
    
    #getting basemap size
    latmin = 0
    latmax = 0
    lonmin = 0
    lonmax = 0

    for i in range(lista.shape[0]):
        if abs(np.nanmin(lista[0][1,:,0]))>lonmin:
            lonmin = np.nanmin(lista[0][1,:,0])
        if abs(np.nanmax(lista[0][1,:,0]))>lonmax:
            lonmax = np.nanmax(lista[0][1,:,0])
        
        if abs(np.nanmin(lista[0][1,:,1]))>latmin:
            latmin = np.nanmin(lista[0][1,:,1])
        if abs(np.nanmax(lista[0][1,:,1]))>latmax:
            latmax = np.nanmax(lista[0][1,:,1])
        
    data = np.array(lista)
    
    
    
    
    
    fig= plt.figure(figsize=(20,6))
    ax = plt.subplot(121,aspect = 'equal')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    m = Basemap(resolution='h'
            , projection='mill'
            , llcrnrlon=lonmin-0.3
            , llcrnrlat=latmin-0.3
            , urcrnrlon=lonmax+0.3
            , urcrnrlat=latmax+0.3,
             )
    meridians = np.arange(-80.,-10,10)
    parallels = np.arange(55, 90, 5)
    m.drawparallels(parallels,labels=[1,0,0,0],color='w', fontsize=10, fontweight='bold', label='_nolegend_')
    meri = m.drawmeridians(meridians,labels=[0,0,0,1],color='w', fontsize=10, fontweight='bold', label='_nolegend_')
    
    print(data.shape[0],' sequences')
    for i in range(data.shape[0]):
        xp, yp = m(data[i][0,:,0], data[i][0,:,1])
        m.scatter(xp, yp, marker='o',s=2,color='b',alpha=1)
        m.plot(xp, yp,color='b',linestyle='-', linewidth=4,alpha=0.6)
        m.scatter(xp[0], yp[0], marker='^',s=60,color='b',edgecolor='black', linewidth=3)
        
        x, y = m(data[i][1,:,0], data[i][1,:,1])
        m.scatter(x, y, marker='o',s=2,color='orange',alpha=1)
        m.plot(x, y,color='orange',linestyle='-', linewidth=4,alpha=0.6)
        m.scatter(x[0], y[0], marker='^',s=60,color='orange',edgecolor='black', linewidth=3)
        
    m.scatter(xp[0], yp[0], marker='^',s=60,color='b',edgecolor='black', linewidth=3,label='Predicted targets')
    m.scatter(x[0], y[0], marker='^',s=60,color='orange',edgecolor='black', linewidth=3,label='True targets')
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    legend = plt.legend(loc='lower left',fontsize=12)
        


    m.drawcoastlines()
    m.bluemarble(scale=4, alpha=0.8)
    ####################    annotating Nuuk
    x_nuuk, y_nuuk = m(-51.7215, 64.18347)
    circle_rad = 15  # This is the radius, in points
    ax.plot(x_nuuk, y_nuuk, 'o',
            ms=circle_rad * 2, mec='k', mfc='none', mew=2)
    ax.annotate('Nuuk', xy=(x_nuuk,y_nuuk), xytext=(60, 60),
                textcoords='offset points',
                color='k', size='large',
                arrowprops=dict(
                    arrowstyle='simple,tail_width=0.15,head_width=0.8,head_length=0.8',
                    facecolor='k', shrinkB=circle_rad * 1.2)
    )




    plt.show()
        
    return None
        
def remove_nan(cleaned_data,cogrow=4):
    '''
    '''
    cleaned_data_no_nan = []
    for i in range(len(cleaned_data)):
        if np.isnan(cleaned_data[i][:,cogrow]).any()==False:
            cleaned_data_no_nan.append(cleaned_data[i])
        
    cleaned_data_no_nan = np.array(cleaned_data_no_nan) 
    return cleaned_data_no_nan
        
def min_seqlength(cleaned_data_no_nan,lookback =30,loockback_multiples=5,verbose=1):
    '''
    '''
    cleaned_data_no_nan_lookback = []
    for i in range(len(cleaned_data_no_nan)):
        if len(cleaned_data_no_nan[i])>(lookback*loockback_multiples):
            cleaned_data_no_nan_lookback.append(cleaned_data_no_nan[i])
        
    cleaned_data_no_nan_lookback = np.array(cleaned_data_no_nan_lookback)  
    
    if verbose>0:
        print(f"Number of sequences: {len(cleaned_data_no_nan_lookback)}. ")
        
    return cleaned_data_no_nan_lookback

import copy

def scale_grid_seq(samples,scaler):
    '''
    '''
    samples_scaled = []
    for i in range(len(samples)):
        samples_scaled.append(scale_data(samples[i],ais_scaler))
        
    samples_scaled = np.array(samples_scaled)
    return samples_scaled



def scale_data_grid(org_data,scaler):
    '''
    SCALING FOR CLASSI
        To prepare the data for the Deep learning methods, it need to be scaled to a standard scale.
        We are here using the standard scaler in the sklearn library, in which the data is scaled to a mean value of 0.
        
        
    '''
    #print(org_data.shape)
    data =copy.deepcopy(org_data)
    if (len(data[0].shape))==3:
        for i in range((data.shape[0])):
            for j in range(data[i].shape[-2]):               
                data[i][:,j,:]=scaler.transform(data[i][:,j,:])
    
    return data


def y_class(change_coord,resolution = 0.01):
    '''
    
    '''    
    y_class= -1
    
    #print(resolution*5)
    
    ########### GRID FOR LOWER CLASSES ##################
    
    if (-resolution <= change_coord[0] <= resolution)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 0')
        y_class = 0
    if (resolution*1 <= change_coord[0] <= resolution*3)  and (resolution <= change_coord[1] <= resolution*3):
        #print('class 1')
        y_class =1
    if (-resolution*1 <= change_coord[0] <= resolution*1)  and (resolution*1 <= change_coord[1] <= resolution*3):
        #print('class 2')
        y_class = 2
    if (-resolution*3 <= change_coord[0] <= -resolution*1) and (resolution*1 <= change_coord[1] <= resolution*3):
        #print('class 3')
        y_class = 3
    if (-resolution*3 <= change_coord[0] <= -resolution*1)   and (-resolution*1 <= change_coord[1] <= resolution*1):
        #print('class 4')
        y_class = 4
    if (-resolution*3 <= change_coord[0] <= -resolution*1)   and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 5')
        y_class = 5
    if (-resolution <= change_coord[0] <= resolution*1)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        ##print('class 6')
        y_class = 6
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 7')
        y_class = 7
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 8')
        y_class = 8
    
        
        
    if (resolution*3 <= change_coord[0] <= resolution*5)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 9')
        y_class = 9
    if (resolution <= change_coord[0] <= resolution*3)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 10')  
        y_class = 10
    if (-resolution <= change_coord[0] <= resolution) and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 11')
        y_class = 11
    if (-resolution*3 <= change_coord[0] <= -resolution)   and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 12')
        y_class = 12
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (resolution*3 <= change_coord[0] <= resolution*5):
        #print('class 13')
        y_class = 13
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (resolution <= change_coord[1] <= resolution*3):
        #print('class 14')
        y_class = 14
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 15')
        y_class = 15
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 16')
        y_class = 16
    if (-resolution*5 <= change_coord[0] <= -resolution*3)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 17')
        y_class = 17
    if (-resolution*3 <= change_coord[0] <= -resolution)   and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 18')
        y_class = 18
    if (-resolution <= change_coord[0] <= resolution)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 19')
        y_class = 19
    if (resolution <= change_coord[0] <= resolution*3)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 20')    
        y_class = 20
    if (resolution*3 <= change_coord[0] <= resolution*5)  and (-resolution*5 <= change_coord[1] <= -resolution*3):
        #print('class 21')
        y_class = 21
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (-resolution*3 <= change_coord[1] <= -resolution*1):
        #print('class 22')
        y_class = 22
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (-resolution <= change_coord[1] <= resolution*1):
        #print('class 23')
        y_class = 23
    if (resolution*3 <= change_coord[0] <= resolution*5)   and (resolution <= change_coord[1] <= resolution*3):
        #print('class 24')
        y_class = 24
        
        
    ########### GRID FOR MIDDLE CLASSES ##################
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 25
    if (resolution <= change_coord[0] <= resolution*5)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 26
    if (-resolution*5 <= change_coord[0] <= 0)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 27
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (resolution*5 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 28
    if (-resolution*10 <= change_coord[0] <= -resolution*5)  and (0 <= change_coord[1] <= resolution*5):
        #print('class 24')
        y_class = 29
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (-resolution*5 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 30
    if (-resolution*10 <= change_coord[0] <= -resolution*5)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 31
    if (-resolution*5 <= change_coord[0] <= 0)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 32
    if (0 <= change_coord[0] <= resolution*5)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 33
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (-resolution*10 <= change_coord[1] <= -resolution*5):
        #print('class 24')
        y_class = 34
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (-resolution*5 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 35
    if (resolution*5 <= change_coord[0] <= resolution*10)   and (0 <= change_coord[1] <= resolution*5):
        #print('class 24')
        y_class = 36
        
        
    ########### GRID FOR advanced CLASSES ##################
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 37
    if (0 <= change_coord[0] <= resolution*10)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 38
    if (-resolution*10 <= change_coord[0] <= 0)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 39
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (resolution*10 <= change_coord[1] <= resolution*20):
        #print('class 24')
        y_class = 40
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (0 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 41
    if (-resolution*20 <= change_coord[0] <= -resolution*10)  and (-resolution*10 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 42
    if (-resolution*20 <= change_coord[0] <= -resolution*10)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 43
    if (-resolution*10 <= change_coord[0] <= 0)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 44
    if (0 <= change_coord[0] <= resolution*10)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 45
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (-resolution*20 <= change_coord[1] <= -resolution*10):
        #print('class 24')
        y_class = 46
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (-resolution*10 <= change_coord[1] <= 0):
        #print('class 24')
        y_class = 47
    if (resolution*10 <= change_coord[0] <= resolution*20)   and (0 <= change_coord[1] <= resolution*10):
        #print('class 24')
        y_class = 48

    ########### GRID FOR greater CLASSES ##################
        
    y_class = y_class+1   
    return y_class

def clean2train(cleaned_data,split_validate_value=0.9,split_traintest=0.7,lookback=25,features=4,fatures_predict=4,temp=True,verbose=0):
    '''
    
    '''
    cleaned_data_1 = add_dist(cleaned_data)
    if temp==True:
        index2=[]
        for i in range(len(cleaned_data_1)):
            if len(np.unique(cleaned_data_1[i][:,1]))<lookback+5:
                index2.append(i)        
        cleaned_data_1 = np.delete(cleaned_data_1, index2)

        cleaned_data_2 = cleaned_data_1.copy()
        for i in range(len(cleaned_data_2)):
            cleaned_data_2[i] = cleaned_data_2[i][:,:]
    if verbose>0:
        print('Shape of shiptype diveded data:\n',cleaned_data_2.shape)
        print('--------------------\nnumber of sequences ',cleaned_data_2.shape)
        print('number of timestamps in first ',len(cleaned_data_2[0]))
        print('Number of timestamps in forth ',len(cleaned_data_2[3]))
        print('Number of features ',len(cleaned_data_2[2][0]))
        
    if verbose>0:
        print('Creating training, testing and validation data\n-----------------------')
        
    X_train, y_train, x_test, y_test, validate_sample, validate_target,scaler = create_dataset(cleaned_data_2,split_validate_value=split_validate_value,split_traintest=split_traintest,lookback=lookback,features=features,fatures_predict=features)

    if verbose>0:
        print('Scaling data\n-----------------------')
    X_train_scaled = scale_data(X_train,scaler)
    y_train_scaled = scale_data(y_train,scaler)
    x_test_scaled = scale_data(x_test,scaler)
    y_test_scaled = scale_data(y_test,scaler)
    
    if verbose>0:
        print('Padding data\n-----------------------')
    max_length = max_lenght(cleaned_data_2)

    x_train_scaled_padded = pad_data(X_train_scaled,max_length)
    y_train_scaled_padded  = pad_data(y_train_scaled,max_length)
    x_test_scaled_padded = pad_data(x_test_scaled,max_length)
    y_test_scaled_padded =pad_data(y_test_scaled,max_length)
    
    return x_train_scaled_padded, y_train_scaled_padded, x_test_scaled_padded, y_test_scaled_padded        
    
    