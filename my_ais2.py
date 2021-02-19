import math
import pickle
import pandas as pd 
import sys
import my_cleaned_ais as ais_prep
import numpy as np

def temp_clean(data,lookback):
    '''
        Delete this later on.
        This is only used for extra cleaning until I figure out exactly how to handle the data
    '''
    
    cleaned_data_1 = data.copy()


    index2=[]
    for i in range(len(data)):
        if len(np.unique(data[i][:,0]))<lookback+5 and (len(np.unique(data[i][:,1]))<lookback+5):
            index2.append(i)        
    cleaned_data_1 = np.delete(cleaned_data_1, index2)


    cleaned_data_1 = ais_prep.add_dist(cleaned_data_1)
    
    cleaned_data_2 = []
    for i in range(len(cleaned_data_1)):
        cleaned_data_2.append(cleaned_data_1[i][:,[0,1,3,-1]])
    
    cleaned_data_2 = np.array(cleaned_data_2)
    
    return cleaned_data_2


def df_to_numpy_training(df_list):
    '''
    Currently:
    removing mmsi and counder for the dataframe, and turning into numpy files.
    '''
    cargo_df_split_resampl_maxlength_corrected = df_list.copy()
    df_cargo_training = []
    for i in range(len(cargo_df_split_resampl_maxlength_corrected)):
        if len(cargo_df_split_resampl_maxlength_corrected[i])>1:
            data = cargo_df_split_resampl_maxlength_corrected[i].drop(['counter'],axis=1)      
            df_cargo_training.append(data.to_numpy())
    
    df_cargo_training = np.array(df_cargo_training)
    return df_cargo_training

def df_splt_smaller_seq(df,max_length=150,verbose=0):
    '''
    
    '''
    cargo_df_split_resampl = df.copy()
    list_df = []
    for i in range(len(cargo_df_split_resampl)):
        temp_df = cargo_df_split_resampl[i]
        lenght_df_temp = len(temp_df)
        if lenght_df_temp>max_length+50:
            number_splits = math.floor(lenght_df_temp/max_length)
            #print(number_splits)
            for j in range(int(number_splits)):
                df_to_append = cargo_df_split_resampl[i].iloc[max_length*j:max_length*(j+1),:]
            
                list_df.append(df_to_append)
            list_df.append(cargo_df_split_resampl[i].iloc[int(number_splits)*max_length:,:])
        else:
            list_df.append(temp_df)
            
    return list_df

def df_correction(list_pd,verbose=0):
    '''
    This function checks for errors in the dataset
        - None values
        - lenght of vectors
        - Value erros
    .
    
    Error correction is here done by manually defined thresholds. It will therefore only be done on the data deemed usefull in the study of this thesis.
    Useful paramters:
        Timmestamp
        Lat: unphyscial values are removed. If there is a wanted area of interest,
            this can be specified here since Ships are expected to behave widly different in different geographical locations 
            (Already sapecifed data from Gatehouse is used and is thus not neccesary). 
        Lon: unphyscial values are removed. If there is a wanted area of interest, this can be specified here. 
        mmsi:
        name
        bearing
        speed: cant be used. Use time and coordinates instead.
        type of ship
        dimensions
        destination
        
    bearning: vessel direction in degrees relative from magnetic north.
    Course Over Ground: The shipts direction relative to absolute north in degrees (i.e noprth pole)
    Speed ober Ground: velovity in nautical miles pr. hour
    
    Every veseel is required to send a signal every  minuts. '
    
    Errors and missing data: see Harati-Mokhtari et al. (2007)
    
    Input: 
        coord_tres[float]: [lat_min,lat_max,lon_min,lon_maks] boudning box.    

    '''
    ##### Part 1: subjective correction #####a##
    #if length of tracks are too short, they are removed.
    #speed: 60 km/h:
    #for i in range(len(df)):
    #    for j in range(df[i]):
    #        if (df[i][j][df[i][j].sog >=par[0]/1.852 ]):
    #            df[i].pop(j)
    #1.852
    #if (coord_tres==lon_thres):
    

            
    #### Part 2:Missing data and Errors ########
    
    list_list_pd = list_pd.copy()
    #correcting data. 
    drop_index=[]
    for j in range(len(list_list_pd)-1,-1,-1): 
        
        # remove if nan exists in cog, lat, lon. 
        # remove if less than 10 datapoints exists.
        # remove if lat is more or less than 180 and -180
        # remove if sog 34
        #        #(any(z for z in list_list_pd[j].iloc[:,4].isnull())) or \
        #        
        if (any(x for x in list_list_pd[j].iloc[:,1].isnull())) or \
        (any(y for y in list_list_pd[j].iloc[:,2].isnull())) or \
        (any(z for z in list_list_pd[j].iloc[:,3].isnull())) or \
        (any(z for z in list_list_pd[j].iloc[:,0].isnull())) or \
        (any(z for z in list_list_pd[j].iloc[:,1]>180)) or \
        (any(z for z in list_list_pd[j].iloc[:,1]<-180)) or \
        (any(z for z in list_list_pd[j].sog>34)):
            drop_index.append(j)
    for k in drop_index:
        list_list_pd.pop(k)
            
        if verbose>0:
            print('(df_resampling):\n',len(drop_index),' datasets has been removed.\n')
                
    return list_list_pd



def df_resampling(df_input,resampling_time=10):
    '''
    Timestamps do not have fixed intervals. 
    It ranges from less than 10 sec to more than hours aparts (amount of hours defined by ais.df_split_ais() )
    To account for this large varibility, and to add more generability in the NN, a resampling is performed.
    This resampling resampel the data using a mean method.
    

    '''
    list_df = df_input.copy()
    list_list_pd = []
    
    for i in range(len(list_df)):
        if len(list_df[i])>0:
            #changing index to time so it can be interpoalted.
            placeholder = list_df[i].copy()
            placeholder.index = list_df[i].iloc[:,1]
            #getting the interpolation for XX min using a linear (mean) approximation
            df_interpol = placeholder.resample(f'{resampling_time}min',origin='start').mean().pad()
            
            #method='spline', order=3
            #adding metadata to dataframe
            #df_interpol.callsign = placeholder.callsign.iloc[0]
            #df_interpol.destination = placeholder.destination.iloc[0]
            #df_interpol.nav_status = placeholder.nav_status.iloc[0]
            #df_interpol.name = placeholder.name.iloc[0]
            #df_interpol.shiptype = placeholder.shiptype.iloc[0]
            list_list_pd.append(df_interpol)

                
        
    return list_list_pd


def df_for_shiptype(df_input,messages_threshold=40, verbose=0):
    '''
        This function takes a df, and make a df for each ship in each shiptype
        messages_threshold[int]: If number of messages for a ship is less than messages_threshold, they will not be used.
    '''
    #type of ships in data
    assert len(df_input.columns)>1,'check your pd columns...'
    df = df_input.copy()
    list_with_df_for_each_ship = []

    individual_mmsi = df.mmsi.unique()
    for i in range(len(individual_mmsi)):
        df_temp = df[df.mmsi ==individual_mmsi[i]]
        if len(df_temp)>messages_threshold:
            #df_temp['index'] = df_temp.columns.get_loc(df.columns[1])
            elapsed = df_temp.iloc[:,1].diff()
        
            if len(elapsed)>0:
                elapsed.iloc[0] = elapsed.iloc[1]-elapsed.iloc[1]
            #when sampling, Timestamp will be cahgned, therefore adding an extra timestamp
            df_temp['timestamp'] = df_temp.iloc[:,1]
            df_temp['elapsed']= elapsed.iloc[:]/ np.timedelta64(60, 's')
            df_temp['counter'] = range(len(df_temp))
            list_with_df_for_each_ship.append(df_temp)
        else:
            if verbose>0:
                print(f"Ship {i} with mmsi {individual_mmsi(i)} has been removed.")

        
    return list_with_df_for_each_ship


def df_split_ais(df_input,allowed_stop_time=340,verbose=0):
    '''
        This function spilt up a AIS df if there is a brek of 
        Since large datasets with long periods will be used, it is important to split up tracks where a ship has been in e.g a habour for a few weeks. 
        There can be multiple of such anchor points. 
        It is the individual tracks that is of interest, and not a ships lifelong journey.
        
        allowed_stop_time[int]: Number of minuts alloowed to break. 
                                if time between messages are larger than allowed_stop_time, the df will be split up.
                                
        If allowed_stop_time is large: only longer stops are used to split up the df. 
        If allowed_stop_time is low: only frequencly recorded tracks will be used. This will reduce the uncertainty when training.
        For long term prediction, a large allowed_stop_time is ok. 
        If the task is to predict short term ship tracks, a small allowed_stop_time is needed.
        
    '''
    list_of_df = df_input.copy()
    list_df_split  =[]
    
    if verbose>0:
        print('there are ',len(list_of_df),'type of ships' )
    for l in range(len(list_of_df)):
        if len(list_of_df[l])>0:
            
            #if elapsed time is lager than 240 min or smaller than 0 min (error or habour)
            position_split = list_of_df[l].query(f"elapsed > {allowed_stop_time} or elapsed < 0").counter.tolist()
            
            #adding the amount of rows to make the first split...
            position_split.append(len(list_of_df[l]))
            #if 'elapsed' in list_of_df[l][i].columns:
            #    del list_of_df[l][i]['elapsed']
            for j in range(len(position_split)-1):
                df_to_append = list_of_df[l].iloc[position_split[-2-j]:position_split[-1-j],:]
                df_to_append.elapsed.iloc[0] = 0
                list_df_split.append(df_to_append)
    return list_df_split





def raw2clean(df_raw,t_messages=100,t_pausetime = 1000,f_resampling = 10,t_maxlengt = 300 ,verbose=0):
    '''
    
    t_messages = 100 # threshold for number of messages
    t_pausetime = 1000 #threshold for pause time
    f_resampling = 10 #resampling frequence
    t_minlengt = 10 # minimum length of sequences.
    t_maxlengt = 300 # maximum length of sequences. sequence time = f_resampling*samples.

    
    '''
    cargo_df = df_for_shiptype(df_raw,t_messages)
    if verbose>0: 
        print(len(cargo_df))
    del df_raw

    cargo_df_split = df_split_ais(cargo_df,t_pausetime)
    if verbose>0: 
        print(len(cargo_df_split))
    del cargo_df

    cargo_df_split_resampl = df_resampling(cargo_df_split,f_resampling)
    if verbose>0: 
        print(len(cargo_df_split_resampl))
    del cargo_df_split

    cargo_df_split_resampl_maxlength = df_splt_smaller_seq(cargo_df_split_resampl,t_maxlengt)
    if verbose>0: 
        print(len(cargo_df_split_resampl_maxlength))
    del cargo_df_split_resampl

    cargo_df_split_resampl_maxlength_corrected = df_correction(cargo_df_split_resampl_maxlength)
    if verbose>0: 
        print(len(cargo_df_split_resampl_maxlength_corrected))
    del cargo_df_split_resampl_maxlength

    testing_set = df_to_numpy_training(cargo_df_split_resampl_maxlength_corrected)
    
    if verbose>0: 
        print(testing_set.shape)
        
    return testing_set,cargo_df_split_resampl_maxlength_corrected


from datetime import date, time, datetime, timedelta
import os


def get_ship_ais(shiptype='cargo',start_date='2019-04-01',end_date='2019-04-02',geo_area='S_greenland_dtu2020',save_df=True,verbose=1):
    '''
    
    '''    
    #getting dates as datetime to convert to countable.
    start_date_count = date(year=int(start_date[0:4]), month=int(start_date[5:7]), day=int(start_date[8:10]))
    end_date_count = date(year=int(end_date[0:4]), month=int(end_date[5:7]), day=int(end_date[8:10]))
    #getting number of days
    n_days = (end_date_count-start_date_count).days
    n_days = n_days+1
    if verbose>0:
        print(f"{n_days} Days are queried.")
    ###############################
    #getting results from start date
    ###############################
    if (shiptype.lower()=='cargo') or (shiptype==70):
        shiptype = 70
    elif shiptype.lower() =='tanker' or (shiptype==80):
        shiptype = 80
    elif shiptype.lower() =='Other Type' or (shiptype==90):
        shiptype = 90
    elif shiptype.lower() =='passenger' or (shiptype==60):
        shiptype = 60
    elif shiptype.lower() =='fishing' or (shiptype==30):
        shiptype = 30
    elif shiptype.lower() =='towing' or (shiptype==31) or (shiptype==32):
        shiptype = 31
    elif shiptype.lower() =='tailing' or (shiptype==36):
        shiptype = 36
    elif shiptype.lower() =='pleasure craft' or (shiptype==37):
        shiptype = 37
    elif shiptype.lower() =='tug' or (shiptype==52):
        shiptype = 52
    elif shiptype.lower() =='law enforcement' or (shiptype==55):
        shiptype = 55
    elif shiptype.lower() =='search and rescue vessel' or (shiptype==51):
        shiptype = 51
    elif shiptype.lower() =='high speed craft' or (shiptype.lower() =='hsc') or (shiptype==40):
        shiptype = 40
    else:
        shiptype = 999
        
    assert shiptype!=999,'Error. ensure that you ship type is one of the allowed types.'
    
        
    #making sql
    #sql = f"\
    #with cte as \
    #(\
    #select mmsi, pgt_pointsm(track) as p from track.tbl_daily where day = '{start_date}' \
    #and mmsi in (select mmsi from dbserver.mat_statvoy where shiptype >= '{shiptype_min}' and shiptype < '{shiptype_max}') \
    #and track && (select polygon from dbserver.tbl_shapes where name = '{geo_area}')\
    #) select mmsi, (p).stamp, st_x((p).pos::geometry), st_y((p).pos::geometry), (p).sog, (p).cog from cte where (p).bits = 1\
    #"
    sql = f"\
    with cte as \
    (\
    select mmsi, pgt_pointsm(track) as p from track.tbl_daily where day = '{start_date}' \
    and mmsi in (select mmsi from dbserver.mat_statvoy where shiptype = '{shiptype}'  \
    and track && (select polygon from dbserver.tbl_shapes where name = '{geo_area}'))\
    ) select mmsi, (p).stamp, st_x((p).pos::geometry), st_y((p).pos::geometry), (p).sog, (p).cog from cte where (p).bits = 1\
    "
    
    if verbose>0:
        print(f"Fetching results from {start_date} ")
    #getting results
    results_startdate = do_one_simple_query(sql)
    #results in dataframe
    df_results = pd.DataFrame(results_startdate, columns =['mmsi', 'datatime', 'lat','lon','sog','cog'])
    
    ###############################
    #getting results from other dates
    ###############################
    if n_days>1:
        for i in range(1,n_days,1):
            current_date = (start_date_count+timedelta(days=i)).strftime("%Y-%m-%d")
            if verbose>0:
                print(f"Fetching results from {current_date} ")
            sql = f"\
            with cte as \
            (\
            select mmsi, pgt_pointsm(track) as p from track.tbl_daily where day = '{current_date}' \
            and mmsi in (select mmsi from dbserver.mat_statvoy where shiptype = '{shiptype}') \
            and track && (select polygon from dbserver.tbl_shapes where name = '{geo_area}')\
            ) select mmsi, (p).stamp, st_x((p).pos::geometry), st_y((p).pos::geometry), (p).sog, (p).cog from cte where (p).bits = 1\
            "
            #fetching results from currect data
            results_current_date = do_one_simple_query(sql)
            #sacing resutls in dataframe
            df_temp = pd.DataFrame(results_current_date, columns =['mmsi', 'datatime', 'lat','lon','sog','cog'])
            #appending curret date results to final dataframe
            df_results = pd.concat([df_results, df_temp], ignore_index=True)
            
    if save_df==True:
        name_df=f"df_{shiptype}_{geo_area}_{start_date}_{end_date}.pkl"
        df_results.to_pickle(name_df)
        size_path = os.path.getsize(f"{name_df}")*9.537*(10**(-7))
        if verbose>0:
            print(f"size of file {round(size_path,4)} Mb")

    if verbose>0:    
        print(f"(func) {len(df_results)} messages are fetched")
    
        
    
    return df_results
            
#plotting
#import gdal
#import osr
#import geopandas as gpd
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

def plot_tracks(lista,projection='mill'):
    '''
    this is only for plotting AIS ship tracks.
    Input is either array or list of ship tracks.
    projeciton:
    lcc
    mill
    '''
        #getting basemap size
    data = np.array(lista)
    latmin = np.nanmin(data[0][:,1])
    latmax = np.nanmax(data[0][:,1])
    lonmin = np.nanmin(data[0][:,0])
    lonmax = np.nanmax(data[0][:,0])
    

    for i in range(data.shape[0]):
        if (np.nanmin(data[i][:,0]))<lonmin:
            lonmin = np.nanmin(data[i][:,0])
        if (np.nanmax(data[i][:,0]))>lonmax:
            lonmax = np.nanmax(data[i][:,0])
        if (np.nanmin(data[i][:,1]))<latmin:
            latmin = np.nanmin(data[i][:,1])
        if (np.nanmax(data[i][:,1]))>latmax:
            latmax = np.nanmax(data[i][:,1])
        
    
    
    
    
    
    
    fig= plt.figure(figsize=(20,6))
    ax = plt.subplot(121,aspect = 'equal')
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0, hspace=0)
    m = Basemap(resolution='h'
            , projection='mill'
            , llcrnrlon=lonmin-(abs(lonmin)*0.1)
            , llcrnrlat=latmin-(abs(latmin)*0.1)
            , urcrnrlon=lonmax+abs(lonmax)*0.1
            , urcrnrlat=latmax+abs(latmax)*0.1,
             )
    meridians = np.arange(-80.,-10,30)
    parallels = np.arange(55, 90, 10)
    m.drawparallels(parallels,labels=[1,0,0,0],color='w', fontsize=10, label='_nolegend_')
    meri = m.drawmeridians(meridians,labels=[0,0,0,1],color='w', fontsize=10, label='_nolegend_')
    
    print(data.shape[0],' sequences')
    for i in range(data.shape[0]):
        xp, yp = m(data[i][:,0], data[i][:,1])
        m.scatter(xp, yp, marker='o',s=2,color='b',alpha=1)
        m.plot(xp, yp,color='b',linestyle='-', linewidth=1,alpha=0.6)
        m.scatter(xp[0], yp[0], marker='^',s=60,color='b',edgecolor='black', linewidth=3)
        
        
    m.scatter(xp[0], yp[0], marker='^',s=60,color='b',edgecolor='black', linewidth=1.5,label='Ship tracks')

    
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