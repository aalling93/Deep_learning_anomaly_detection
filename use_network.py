from sklearn.manifold import TSNE
import matplotlib.cm as cm
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import pickle

def plot_loss(vae_model,encoder,history,beta):
    '''training_data_for_reshape_scaled_nommsi,original_data
    '''
    #with open('history_epoch_20_VAE_VSAM_model_20201208-1626_20201208-1947.pkl', 'rb') as fh:
    #        history = pickle.load(fh)
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    color = 'tab:blue'
    try:
        ax1.plot(history.history['vae_kl_loss'], linewidth=2,color='tab:blue',label='KL loss')#kullback_leibler_divergence
        ax1.plot(history.history['val_vae_kl_loss'], linewidth=2,color='tab:green',label='val KL loss') 
        if (np.max(history.history['vae_kl_loss'])-np.min(history.history['vae_kl_loss']))>100:
            ax1.set_yscale('log') 
    except:
        ax1.plot(history.vae_kl_loss, linewidth=2,color='tab:blue',label='KL loss')
        ax1.plot(history['val_vae_kl_loss'], linewidth=2,color='tab:green',label='val KL loss')
        if (np.max(history['vae_kl_loss'])-np.min(history['vae_kl_loss']))>100:
            ax1.set_yscale('log') 
        
    #plt.title('With attention')
    plt.ylabel('KL loss ')
    plt.xlabel('Epochs', fontsize=22)
    plt.title('')
    ax1.set_ylabel('KL loss',color='tab:blue', fontsize=22)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
    plt.legend(loc="center left", fontsize=20)
    
       
        
    ax2 = ax1.twinx()
    try:
        ax2.plot(history.history['vae_reconstruction_loss'], linewidth=2,color='tab:orange',label='reconstruction loss')
        ax2.plot(history.history['val_vae_reconstruction_loss'], linewidth=2,color='tab:red',label='val reconstruction loss')
        if (np.max(history.history['vae_reconstruction_loss'])-np.min(history.history['vae_reconstruction_loss']))>100:
            ax2.set_yscale('log')   
    except:
        ax2.plot(history['vae_reconstruction_loss'], linewidth=2,color='tab:orange',label='reconstruction loss')
        ax2.plot(history['val_vae_reconstruction_loss'], linewidth=2,color='tab:red',label='val reconstruction loss')
        if (np.max(history['vae_reconstruction_loss'])-np.min(history['vae_reconstruction_loss']))>100:
            ax2.set_yscale('log') 
        
    plt.legend(loc="center right", fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=22)
    #plt.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_ylabel('Reconstruction loss ', color='orange', fontsize=22)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=20)  
    plt.xticks(fontsize=15)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    
    plt.savefig(f'{vae_model.name}/vae_kl_loss_vae_reconstruction_loss.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    plt.show()
    fig, ax1 = plt.subplots(figsize=(10,7))
    color = 'black'
    try:
        ax1.plot(history.history['loss'], linewidth=2,color='tab:blue',label='Training loss')#kullback_leibler_divergence
        ax1.plot(history.history['val_loss'], linewidth=2,color='tab:green',label='Val loss')  
        if (np.max(history.history['loss'])-np.min(history.history['loss']))>100:
            ax1.set_yscale('log') 
    except:
        ax1.plot(history['loss'], linewidth=2,color='tab:blue',label='Training loss')#kullback_leibler_divergence
        ax1.plot(history['val_loss'], linewidth=2,color='tab:green',label='Val loss')  
        if (np.max(history['loss'])-np.min(history['loss']))>100:
            ax1.set_yscale('log') 
        
    plt.title('With attention')
    plt.ylabel('Training')
    plt.xlabel('Epochs', fontsize=20)
    plt.title('')
    ax1.set_ylabel('Loss',color='black', fontsize=22)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
    plt.legend(loc="center left", fontsize=20)
      
      
        
    #ax2 = ax1.twinx()
    #try:
    #    ax2.plot(history.history['val_cosine_similarity'],color='tab:red',label='Valitdation cosine similarity')
    #    ax2.plot(history.history['cosine_similarity'],color='tab:orange',label='Cosine similarity')
    #    if (np.max(history.history['val_cosine_similarity'])-np.min(history.history['val_cosine_similarity']))>100:
    #        ax2.set_yscale('log') 
    #except:
    #    ax2.plot(history['val_cosine_similarity'],color='tab:red',label='Valitdation cosine similarity')
    #    ax2.plot(history['cosine_similarity'],color='tab:orange',label='Cosine similarity')
    #    if (np.max(history['val_cosine_similarity'])-np.min(history['val_cosine_similarity']))>100:
    #        ax2.set_yscale('log') 
    #        
    #plt.legend(loc="center right", fontsize=18)
    #ax2.xaxis.set_tick_params(labelsize=20)
    ##plt.tick_params(axis='x', which='major', labelsize=12)
    #ax2.set_ylabel('Validation Loss', color='orange', fontsize=20)
    #ax2.tick_params(axis='y', labelcolor='orange', labelsize=15)
    # 
    plt.xticks(fontsize=20)
    #ax2.xaxis.set_tick_params(labelsize=18)
    ax1.xaxis.set_tick_params(labelsize=20)
    
    plt.savefig(f'{vae_model.name}/val_cosine_similarity_loss.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    plt.show()
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    color = 'tab:blue'
    try:
        ax1.plot(history.history['lr'], linewidth=2,color='tab:blue',label='Learning Rate')#kullback_leibler_divergence     
    except:
        ax1.plot(history['lr'], linewidth=2,color='tab:blue',label='Learning Rate warm up')#kullback_leibler_divergence 
    #plt.title('With attention')
    #plt.ylabel('Training')
    plt.xlabel('Epochs', fontsize=22)
    plt.title('')
    ax1.set_ylabel('Learning Rate',color='tab:blue', fontsize=22)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
    plt.legend(loc="center left", fontsize=20)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=20)
    
    ax2 = ax1.twinx()
    try:
        ax2.plot(history.history['loss'], linewidth=2,color='tab:orange',label='Training loss')
        ax2.plot(history.history['val_loss'], linewidth=2,color='tab:green',label='Val loss')  
        if (np.max(history.history['loss'])-np.min(history.history['loss']))>100:
            ax2.set_yscale('log') 
    except:
        ax2.plot(history['loss'], linewidth=2,color='tab:orange',label='loss')
        if (np.max(history['loss'])-np.min(history['loss']))>100:
            ax2.set_yscale('log') 
    plt.legend(loc="center right", fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=22)
    #plt.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_ylabel('Loss ', color='orange', fontsize=22)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=20)
    ax2.tick_params(axis='x', labelcolor='black', labelsize=20)
    plt.xticks(fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    
    plt.savefig(f'{vae_model.name}/loss_lr.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    plt.show()
    #plt.savefig(f'{vae_model.name}/vae_loss.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    
    
    fig, ax1 = plt.subplots(figsize=(10,7))
    color = 'tab:blue'
    try:
        ax1.plot(history.history['lr'], linewidth=2,color='tab:blue',label='Learning Rate warm up')#kullback_leibler_divergence    
    except:
        ax1.plot(history['lr'], linewidth=2,color='tab:blue',label='Learning Rate warm up')#kullback_leibler_divergence   
        
    #plt.title('With attention')
    plt.ylabel('Learning rate')
    plt.xlabel('Epochs', fontsize=22)
    plt.title('')
    ax1.set_ylabel('Learning Rate',color='tab:blue', fontsize=22)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
    plt.legend(loc="center left", fontsize=20)
    
    
    ax2 = ax1.twinx()
    ax2.plot(beta, linewidth=2,color='tab:orange',label='KL warm up')
    plt.legend(loc="center right", fontsize=20)
    ax2.xaxis.set_tick_params(labelsize=22)
    #plt.tick_params(axis='x', which='major', labelsize=12)
    ax2.set_ylabel('KL regularisation ', color='orange', fontsize=22)
    ax2.tick_params(axis='y', labelcolor='orange', labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)
    ax1.xaxis.set_tick_params(labelsize=20)
    
    plt.savefig(f'{vae_model.name}/beta_warm_up_lr.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    plt.show()

    #plt.savefig(f'{vae_model.name}/kl_reconstruction_loss.png',transparent=True,bbox_inches='tight',pad_inches =0.1)
    #plt.xticks(fontsize=15)
    #plt.show()
    
    return None



import dask
from sklearn.manifold import TSNE
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
import matplotlib.cm as cm
import matplotlib as mpl
from sklearn.cluster import DBSCAN
from bioinfokit.visuz import cluster

def get_results_2(vae_model,encoder,training_data_for_reshape_scaled_nommsi,original_data,verbose=1):
    '''
    
    '''
    
    if verbose>0:
        print('getting predicitons encoder')
    latent_vector_mean_data,latent_vector_variance_data,latent_vector_data = encoder.predict(training_data_for_reshape_scaled_nommsi)
    np.save(f'{vae_model.name}/latent_vector_data.npy', latent_vector_data)
    np.save(f'{vae_model.name}/latent_vector_mean_data.npy', latent_vector_mean_data)
    np.save(f'{vae_model.name}/latent_vector_variance_data.npy', latent_vector_variance_data)
    
    
    
    ##### Getting classes ###############
    
    print('getting embeddings')
    space_latent_vector_data = TSNE(n_components=2).fit_transform(latent_vector_data)
    space_latent_vector_mean_data = TSNE(n_components=2).fit_transform(latent_vector_mean_data)
    space_latent_vector_variance_data = TSNE(n_components=2).fit_transform(latent_vector_variance_data)
    np.save(f'{vae_model.name}/space_vector_data.npy', latent_vector_data)
    np.save(f'{vae_model.name}/space_space_latent_vector_mean_dataa.npy', space_latent_vector_mean_data)
    np.save(f'{vae_model.name}/space_space_latent_vector_variance_dataa.npy', space_latent_vector_variance_data)
    ##############################################
    
    ##### Getting predictions ###############
    
    print('getting predicitons vae')
    pred = vae_model.predict(training_data_for_reshape_scaled_nommsi)
    #mse = np.sum( (pred-samples_scaled_padded_reshaped)**2 )/(pred-samples_scaled_padded_reshaped).shape[1]
    np.save(f'{vae_model.name}/predictions_vector_data.npy', pred)
    ##############################################
    ##### Getting all the MSE   ###############
    
    print('getting MSE')
    mse_all = []
    lengt = (pred-training_data_for_reshape_scaled_nommsi[0:len(pred)]).shape[0]
    for i in range(lengt):
        mse_all.append(np.sum( (pred[i]-training_data_for_reshape_scaled_nommsi[i])**2 )/(lengt))

    mse_all = np.array(mse_all)
    np.save(f'{vae_model.name}/mse_data.npy', mse_all)
    ##############################################
    ##### Getting the highest and lowest MSE ###############
    print('getting highest and lowest MSE')
    highest_values = (-mse_all).argsort()[:100]
    lowest_values = (-mse_all).argsort()[-100:]
    y = np.linspace(0,len(mse_all),len(mse_all))
    #https://reneshbedre.github.io/blog/tsne.html
    #https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    

    
        
        
    return [latent_vector_mean_data,latent_vector_variance_data,latent_vector_data],[space_latent_vector_data,space_latent_vector_mean_data,space_latent_vector_variance_data],pred, [mse_all,lengt,highest_values,lowest_values,y]



def get_classes(vae_model,space,get_clusters,eps=3,min_samples=200,leaf_size=1000):
    '''
    '''
    ##############################################
    ############# getting clusters #################
    print('getting getting clusters')
    get_clusters = DBSCAN(eps=eps, min_samples=min_samples,leaf_size=leaf_size).fit_predict(space)
    
    ##############################################
    ############# appending cluisters to embedding #################
    print('stacking clusters')
    space_clusters = np.hstack((space,get_clusters[:,None]))
    
    ##############################################
    ############# making lists for each cluster with mmsi #################
    
    index_clusters = []
    mmsi_not_in_cluster = []
    print('indexing trajectories in clusters')
    for i in np.unique(get_clusters):
        index_clusters.append(np.where(get_clusters == i)[0])
    
    open_file = open(f'{vae_model.name}/classes_clustering_DBSCAN_indexofmmsi.pkl', "wb")
    pickle.dump(index_clusters, open_file)
    open_file.close()
    
    return get_clusters, space_clusters,index_clusters