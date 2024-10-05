# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # RUN THIS COMMAND IN ANOTHER NOTEBOOK  :
#
# # ### %run [path]\JDCOOT_FUNCTIONS.ipynb 

# # Import and Utils

# +
# #!pip install ipynb
# #!pip install tf_keras --user
# #!pip install pot
# #!pip install tensorflow
# #!pip install --upgrade keras-preprocessing
# #!pip install scipy
# #!pip install matplotlib --upgrade --user
# #!pip install alive_progress
# #!pip install nodejs

# #!pip install langchain==0.1.6 
# #!pip install langchain-community==0.0.19 
# #!pip install langchain-core==0.1.23
# #!pip install nvm
# #!nvm use 12.5.0
# #!jupyter labextension install @jupyter-widgets/jupyterlab-manager
# #!jupyter nbextension enable --py widgetsnbextension
# #!pip install ipympl
# @title IMPORT USEFULL PACKAGES

# BASICS
import numpy as np # scientific computing
import math
import pandas as pd
import scipy as sp # Fundamental algorithms for scientific computing
from scipy import ndimage # image processing
import scipy.optimize as spo
from scipy.spatial.distance import cdist
from scipy.io import loadmat # files .mat
import scipy.io # Input and output
from scipy import stats # Statistical functions
from scipy.sparse import random # Generate a sparse matrix of the given shape and density with randomly distributed values

# %matplotlib inline
import matplotlib.pyplot  as plt # Visualization // Marion: matplotlib.pylab
import matplotlib.colors as mcolors
from matplotlib import cm
import os # operating system / read/write files
import time # handling time-related tasks
from random import shuffle #pseudo-random number generators for various distributions

from tensorflow.python.ops.numpy_ops import np_config # warning ds import ot

# OT
import ot # OT methods, from install pot

# NN
#import keras # for DL, from install tensorflow
#from keras import layers
import tf_keras
from tf_keras import layers
from tf_keras.models import Sequential
from tf_keras.layers import Activation # Marion : from keras.layers.core ?
from tf_keras.layers import Dense # Marion : from keras.layers.core ?
from tf_keras import backend as K

# ML
import sklearn # for ML
from sklearn import preprocessing
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel

# MISC
from functools import reduce #  for using higher-order functions that are in-built in


from alive_progress import alive_bar
import os
import json
# PERSO
#import cot # COOT
#import jdcot # JDCOOT
#import classif
pd.options.mode.chained_assignment = None  # default='warn'
# -

global INDEX_GENERATION
INDEX_GENERATION=np.random.choice(np.arange(100), math.ceil(0.75 * 100),replace=False)

# +
"""
Store(Data,name) : Store Data from a scenario

Input : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        MC_coot_perf,
                                                                        MC_jdcoot_perf,
                                                                        MC_ref_perf,
                                                                        bp_coot,
                                                                        bp_jdcoot,
                                                                        test_MC_coot_perf,
                                                                        test_MC_jdcoot_perf,
                                                                        test_MC_ref_perf,
                                                                        test_bp_coot,
                                                                        test_bp_jdcoot
        name (str) : Name of the simulation to give in the folder (only the name, not the path)
""" 

def Store(Data,name):
    if not os.path.isdir(os.getcwd()+"\\Stored_Data\\"):
        os.mkdir(os.getcwd()+"\\Stored_Data\\")
        
    if os.path.isdir(os.getcwd()+"\\Stored_Data\\"+name+"\\") : 
        for i in range(1,1000):
            if not os.path.isdir(os.getcwd()+"\\Stored_Data\\"+name+"_"+str(i)+"\\") :
                name=name+"_"+str(i)
                print("Existing save, creating " + name)
                break
        
    
    os.mkdir(os.getcwd()+"\\Stored_Data\\"+name+"\\")
    for i in range(len(Data)):
        np.save(os.getcwd()+"\\Stored_Data\\"+name+"\\"+str(i), Data[i], allow_pickle=True, fix_imports=True)
    print("Saved as : " + os.getcwd()+"\\Stored_Data\\"+name)
            

"""
Read(name) : Read Data from an existing save

Input : name (str) : Name of the simulation given in the folder (only the name, not the path)

Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
                                                                        
""" 
def Read(name):
    if not os.path.isdir(os.getcwd()+"\\Stored_Data\\"+name+"\\") : 
        print("Not existing save")
        return 0
    else :
        Data=[]
        for i in range(len(os.listdir(os.getcwd()+"\\Stored_Data\\"+name+"\\"))):
            Data.append(np.load(os.getcwd()+"\\Stored_Data\\"+name+"\\"+str(i)+".npy"))   
    return tuple(Data)


# +
"""
graph_multi : to plot graph, called in other functions
"""

def graph_multi(Data,algo,TAB,Objective_Variable,maintit,metric,name_metric):
    Proportion_Labelled_Variations=TAB
    Sample_sizes=TAB
    
    
    MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
    
    
    if Objective_Variable == "discrete":
        tit2="Accuracy"
    else : tit2 ="MSE"
    
    
    if True :
    
        if algo=='COOT' : 
            
                fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
                fig.suptitle(maintit,fontsize=15)
                tit="Test " + tit2  
                axs[1,0].set_title(tit)
                axs[1,0].set_ylabel(tit2)
                axs[1,0].set_xlabel(metric)
                axs[1,0].plot(np.arange(len(Proportion_Labelled_Variations))+1,test_MC_coot_perf,color="red",label="COOT")
                axs[1,0].boxplot([test_bp_coot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
               boxprops = dict(facecolor = "red"))

                tit="Pure " + tit2 
                axs[0,0].set_title(tit)
                axs[0,0].set_ylabel(tit2)
                axs[0,0].set_xlabel(metric)
                axs[0,0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT")
                axs[0,0].boxplot([bp_coot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
               boxprops = dict(facecolor = "red"))

                
                min_ =min(np.min(hm_coot),np.min(test_hm_coot)) 
                max_=max(np.max(hm_coot),np.max(test_hm_coot))
            
                tit="HM for COOT (Pure)" 
                axs[0,1].set_title(tit)
                axs[0,1].set_xlabel("Target data " + name_metric)
                axs[0,1].set_ylabel("Source data " + name_metric)
                c1=axs[0,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),hm_coot,vmin=min_,vmax=max_,cmap="viridis")
                axs[0,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
                axs[0,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
                fig.colorbar(c1, ax=axs[0,1])

                tit="HM for COOT (test)" 
                axs[1,1].set_title(tit)
                axs[1,1].set_xlabel("Target data " + name_metric)
                axs[1,1].set_ylabel("Source data " + name_metric)
                axs[1,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),test_hm_coot,vmin=min_,vmax=max_,cmap="viridis")
                axs[1,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
                axs[1,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
                fig.colorbar(c1, ax=axs[1,1])
                
                fs="6"
                

        if algo=='JDCOOT' : 
            
            
            fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
            fig.suptitle(maintit,fontsize=15)
            
            tit="Test " + tit2  
            axs[1,0].set_title(tit)
            axs[1,0].set_ylabel(tit2)
            axs[1,0].set_xlabel(metric)
            axs[1,0].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT")
            axs[1,0].boxplot([test_bp_jdcoot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure " + tit2 
            axs[0,0].set_title(tit)
            axs[0,0].set_ylabel(tit2)
            axs[0,0].set_xlabel(metric)
            axs[0,0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT")
            axs[0,0].boxplot([bp_jdcoot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))


            min_ =min(np.min(hm_jdcoot),np.min(test_hm_jdcoot)) 
            max_=max(np.max(hm_jdcoot),np.max(test_hm_jdcoot))
            
            tit="HM for JDCOOT (pure)" 
            axs[0,1].set_title(tit)
            axs[0,1].set_xlabel("Target data " + name_metric)
            axs[0,1].set_ylabel("Source data " + name_metric)
            c1=axs[0,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),hm_jdcoot,vmin=min_,vmax=max_,cmap="viridis")
            axs[0,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[0,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0,1])
            
            tit="HM for JDCOOT (test)" 
            axs[1,1].set_title(tit)
            axs[1,1].set_xlabel("Target data " + name_metric)
            axs[1,1].set_ylabel("Source data " + name_metric)
            axs[1,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),test_hm_jdcoot,vmin=min_,vmax=max_,cmap="viridis")
            axs[1,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[1,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1,1])
            
            fs="6"
            
            


        if algo == 'both' :
            
            fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
            fig.suptitle(maintit,fontsize=15)
            
            
            axs[1,0].set_ylabel(tit2)
            axs[1,0].set_xlabel(metric)
            axs[1,0].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT")
            axs[1,0].boxplot([test_bp_coot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            axs[0,0].set_ylabel(tit2)
            axs[0,0].set_xlabel(metric)
            axs[0,0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT")
            axs[0,0].boxplot([bp_coot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            
           

            tit="Test " + tit2 
            axs[1,0].set_title(tit)
            axs[1,0].set_ylabel(tit2)
            axs[1,0].set_xlabel(metric)
            axs[1,0].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT")
            axs[1,0].boxplot([test_bp_jdcoot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure " + tit2  
            axs[0,0].set_title(tit)
            axs[0,0].set_ylabel(tit2)
            axs[0,0].set_xlabel(metric)
            axs[0,0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT")
            axs[0,0].boxplot([bp_jdcoot.reshape((-1,len(Sample_sizes)))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            
            min_ =min(np.min(hm_jdcoot),np.min(hm_coot)) 
            max_=max(np.max(hm_jdcoot),np.max(hm_coot))
            min_test =min(np.min(test_hm_jdcoot),np.min(test_hm_coot)) 
            max_test=max(np.max(test_hm_jdcoot),np.max(test_hm_coot))
            
            Mi=min(min_,min_test)
            Ma=max(max_,max_test)
            
            tit="HM for COOT (Pure)" 
            axs[0,1].set_title(tit)
            axs[0,1].set_xlabel("Target data " + name_metric)
            axs[0,1].set_ylabel("Source data " + name_metric)
            c1=axs[0,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),hm_coot,vmin=Mi, vmax=Ma,cmap="viridis")
            axs[0,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str),rotation=45)
            axs[0,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0,1])


            tit="HM for JDCOOT (Pure)" 
            axs[0,2].set_title(tit)
            axs[0,2].set_xlabel("Target data " + name_metric)
            axs[0,2].set_ylabel("Source data " + name_metric)
            axs[0,2].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),hm_jdcoot,vmin=Mi, vmax=Ma,cmap="viridis")

            axs[0,2].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str),rotation=45)
            axs[0,2].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0,2])
            
          
            
 
            
            tit="HM for JDCOOT (Test)" 
            axs[1,2].set_title(tit)
            axs[1,2].set_xlabel("Target data " + name_metric)
            axs[1,2].set_ylabel("Source data " + name_metric)
            c2=axs[1,2].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),test_hm_jdcoot,vmin=Mi, vmax=Ma,cmap="viridis")
            axs[1,2].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str),rotation=45)
            axs[1,2].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1,2])
            
            tit="HM for COOT (Test)" 
            axs[1,1].set_title(tit)
            axs[1,1].set_xlabel("Target data " + name_metric)
            axs[1,1].set_ylabel("Source data " + name_metric)
            axs[1,1].pcolormesh(np.arange(len(Sample_sizes)),np.arange(len(Sample_sizes)),test_hm_coot,vmin=Mi, vmax=Ma,cmap="viridis")
            axs[1,1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str),rotation=45)
            axs[1,1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1,1])
            
            fs="4"
            
            
            


        axs[0,0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1,0].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        
        #axs[1,0].legend(loc="best",fontsize="7")
        axs[0,0].legend(loc="best",fontsize=fs)
        axs[0,0].sharex(axs[1,0])
        axs[0,0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str),rotation=45)
        axs[1,0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str),rotation=45)
        fig.tight_layout()
        #fig.show()
    


# -

# # Algorithms

# +
#@title USEFULL FUNCTIONS

def loss_crossentropy(Y, F):
    eps = 1e-12
    res = np.zeros((Y.shape[0], F.shape[0]))
    logF = np.array(K.log(F + eps))
    for i in range(Y.shape[1]):
        res += -Y[:, i].reshape((Y.shape[0], 1)) * logF[:, i].reshape((1, F.shape[0]))
        
    return res.numpy()


def loss_crossentropy2(Y, F):
    eps = 1e-12
    Flog = K.log(F + eps)
    # loss calculation based on double sum (sum_ij (ys^i, ypred_t^j))
    res = -K.dot(K.variable(Y), K.transpose(Flog))
    
    return res.numpy()


def loss_hinge(Y, F):
    res = np.zeros((Y.shape[0], F.shape[0]))
    for i in range(Y.shape[1]):
        res += (
            np.maximum(
                0,
                1 - Y[:, i].reshape((Y.shape[0], 1)) * F[:, i].reshape((1, F.shape[0])),
            )
            ** 2
        )
    return res.numpy()

def comp_(v=1e6):
    def comp(x,y):
        if x==y or y==-1:
            return 0
        else:
            return v
    return comp

def comp_regression():
    def comp(x,y):
        if x==y or np.isnan(y):
            return 0
        else:
            return (x-y)**2 #MSE ou np.abs()
    return comp

# @title BREGMAN

# import numpy as np

def sinkhorn_scaling(a,b,K,numItermax=1000, stopThr=1e-9, verbose=False,log=False,always_raise=False, **kwargs):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if nbb:
        u = np.ones((Nini, nbb)) / Nini
        v = np.ones((Nfin, nbb)) / Nfin
    else:
        u = np.ones(Nini) / Nini
        v = np.ones(Nfin) / Nfin

    # print(reg)
    # print(np.min(K))

    Kp = (1 / a).reshape(-1, 1) * K
    cpt = 0
    err = 1
    while (err > stopThr and cpt < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        zero_in_transp=np.any(KtransposeU == 0)
        nan_in_dual= np.any(np.isnan(u)) or np.any(np.isnan(v))
        inf_in_dual=np.any(np.isinf(u)) or np.any(np.isinf(v))
        if zero_in_transp or nan_in_dual or inf_in_dual:
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration in sinkhorn_scaling', cpt)
            #if zero_in_transp:
                #print('Zero in transp : ',KtransposeU)
            #if nan_in_dual:
                #print('Nan in dual')
                #print('u : ',u)
                #print('v : ',v)
                #print('KtransposeU ',KtransposeU)
                #print('K ',K)
                #print('M ',M)

            #    if always_raise:
            #        raise NanInDualError
            #if inf_in_dual:
            #    print('Inf in dual')
            u = uprev
            v = vprev

            break
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
    if log:
        log['u'] = u
        log['v'] = v

    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(
                u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))
        
        
# @title COOT

#import numpy as np
#import ot
#from scipy import stats
#from scipy.sparse import random
#from bregman import sinkhorn_scaling

def random_gamma_init(p,q, **kwargs):
    """ Returns random coupling matrix with marginal p,q
    """
    rvs=stats.beta(1e-1,1e-1).rvs
    S=random(len(p), len(q), density=1, data_rvs=rvs)
    return sinkhorn_scaling(p,q,S.A, **kwargs)

def init_matrix_np(X1, X2, v1, v2):
    """Return loss matrices and tensors for COOT fast computation
    Returns the value of |X1-X2|^{2} \otimes T as done in [1] based on [2] for the Gromov-Wasserstein distance.
    Where :
        - X1 : The source dataset of shape (n,d)
        - X2 : The target dataset of shape (n',d')
        - v1 ,v2 : weights (histograms) on the columns of resp. X1 and X2
        - T : Coupling matrix of shape (n,n')
    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2.

    Returns
    -------
    constC : ndarray, shape (n, n')
        Constant C matrix (see paragraph 1.2 of supplementary material in [1])
    hC1 : ndarray, shape (n, d)
        h1(X1) matrix (see paragraph 1.2 of supplementary material in [1])
    hC2 : ndarray, shape (n', d')
        h2(X2) matrix (see paragraph 1.2 of supplementary material in [1])
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    .. [2] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    def f1(a):
        return (a ** 2)

    def f2(b):
        return (b ** 2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
                     np.ones(f1(X2).shape[0]).reshape(1, -1))
    constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
                     np.dot(v2.reshape(1, -1), f2(X2).T))

    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)

    return constC, hX1, hX2


def cot_numpy(X1, X2, w1 = None, w2 = None, v1 = None, v2 = None,
              niter=10, algo='emd', reg=0,algo2='emd',
              reg2=0, verbose=True, log=False, random_init=False, C_lin=None):

    """ Returns COOT between two datasets X1,X2 (see [1])

    The function solves the following optimization problem:
    .. math::
        COOT = \min_{Ts,Tv} \sum_{i,j,k,l} |X1_{i,k}-X2_{j,l}|^{2}*Ts_{i,j}*Tv_{k,l}

    Where :
    - X1 : The source dataset
    - X2 : The target dataset
    - w1,w2  : weights (histograms) on the samples (rows) of resp. X1 and X2
    - v1,v2  : weights (histograms) on the features (columns) of resp. X1 and X2

    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    w1 : numpy array, shape (n,)
        Weight (histogram) on the samples of X1. If None uniform distribution is considered.
    w2 : numpy array, shape (n',)
        Weight (histogram) on the samples of X2. If None uniform distribution is considered.
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1. If None uniform distribution is considered.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2. If None uniform distribution is considered.
    niter : integer
            Number max of iterations of the BCD for solving COOT.
    algo : string
            Choice of algorithm for solving OT problems on samples each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    algo2 : string
            Choice of algorithm for solving OT problems on features each iteration. Choice ['emd','sinkhorn'].
            If 'emd' returns sparse solution
            If 'sinkhorn' returns regularized solution
    reg : float
            Regularization parameter for samples coupling matrix. Ignored if algo='emd'
    reg2 : float
            Regularization parameter for features coupling matrix. Ignored if algo='emd'
    eps : float
        Threshold for the convergence
    random_init : bool
            Wether to use random initialization for the coupling matrices. If false identity couplings are considered.
    log : bool, optional
         record log if True
    C_lin : numpy array, shape (n, n')
            Prior on the sample correspondences. Added to the cost for the samples transport

    Returns
    -------
    Ts : numpy array, shape (n,n')
           Optimal Transport coupling between the samples
    Tv : numpy array, shape (d,d')
           Optimal Transport coupling between the features
    cost : float
            Optimization value after convergence
    log : dict
        convergence information and coupling marices
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    Example
    ----------
    import numpy as np
    from cot import cot_numpy

    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    cot_numpy(Xs,Xt)
    """
    if v1 is None:
        v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
        v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    if w1 is None:
        w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
        w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    if not random_init:
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (X1.shape[0] * X2.shape[0])  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (X1.shape[1] * X2.shape[1])  # is (d,d')
    else:
        Ts=random_gamma_init(w1,w2)
        Tv=random_gamma_init(v1,v2)


    constC_s, hC1_s, hC2_s = init_matrix_np(X1, X2, v1, v2)

    constC_v, hC1_v, hC2_v = init_matrix_np(X1.T, X2.T, w1, w2)
    cost = np.inf

    log_out ={}
    log_out['cost'] = []

    for i in range(niter):
        Tsold = Ts
        Tvold = Tv
        costold = cost

        M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
        if C_lin is not None:
            M=M+C_lin
        if algo == 'emd':
            Ts = ot.emd(w1, w2, M, numItermax=1e7)
        elif algo == 'sinkhorn':
            Ts = ot.sinkhorn(w1, w2, M, reg)

        M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)

        if algo2 == 'emd':
            Tv = ot.emd(v1, v2, M, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Tv = ot.sinkhorn(v1,v2, M, reg2)

        delta = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
        cost = np.sum(M * Tv)

        if log:
            log_out['cost'].append(cost)

        if verbose:
            print('Delta: {0}  Loss: {1}'.format(delta, cost))

        if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print('converged at iter ', i)
            break
    if log:
        return Ts, Tv, cost, log_out
    else:
        return Ts, Tv, cost


# @title JDCOOT

# XA : source domain data - is (nA,dA)
# yA : source labels
# XB : target domain data is (nB,dB)
# yB is optionnal, target labels just to measure performances of the method along iterations
# wA : weights of source samples (default=None which means uniform distribution if not specified)
# wB : weights of target samples (default=None which means uniform distribution if not specified)
# vA : weights of source features (default=None which means uniform distribution if not specified)
# vB : weights of target features (default=None which means uniform distribution if not specified)
# gamma: RBF kernel param (default = 1)
# numIterBCD: number of Iterations for BCD (default = 10)
# alpha: ponderation between ground cost + function cost
# algo/algo2 : choice of algorithm for transport computation for respectively the samples and features transport (default="emd")
# reg/reg2 : choice of the regularization parameter if algo/algo2 is "sinkhorn" (default = 0)
# random_init : wether the transportation plans are randomly initiated (default=False)
 
    

def jdcot_svm(XA,yA,XB,yB=[],wA=None,wB=None,vA=None,vB=None, 
              algo='emd', reg=0,algo2='emd',reg2=0,
              gamma_g=1, numIterBCD = 10, alpha=1,
              lambd=1e1, ktype='linear',random_init=False):
    
    # Initializations
    nA,dA = XA.shape
    nB,dB = XB.shape
    
    if vA is None:
        vA = np.ones(dA) / dA  # is (dA,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (dB,)
    if wA is None:
        wA = np.ones(nA) / nA  # is (nA,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (nB,)

    if not random_init:
        Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
        Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
    else:
        Gs=random_gamma_init(wA,wB) 
        Gv=random_gamma_init(vA,vB)

    # original loss 
    C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)
    cost = np.inf
    
    Gsold = Gs
    Gvold = Gv
    costold = cost

    # classifier    
    g = classif.SVMClassifier(lambd)

    # compute kernels
    if ktype=='rbf':
        Kt=sklearn.metrics.pairwise.rbf_kernel(XB,gamma=gamma_g)
        #Ks=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_g)
    else:
        Kt=sklearn.metrics.pairwise.linear_kernel(XB)
        #Ks=sklearn.metrics.pairwise.linear_kernel(X)
        
    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}
    ypred=np.zeros(yA.shape)
    Chinge=np.zeros((nA,nB))
    
    # do it only if the final labels were given
    if len(yB):
        TBR.append(np.mean(yB==np.argmax(ypred,1)))

    k=0
    while (k<numIterBCD):
       
        k+=1
        # step 1 : samples coupling optimization 
        Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T))+Chinge # is (n,n')          
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (d,d')
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA,vB, Mv, reg2)


        if k>1:
            sav_fcost.append(np.sum(Gs*Chinge))
            sav_totalcost.append(np.sum(Gs*Ms))

        
        # step 3 : prediction function optimization
        Yst=nB*Gs.T.dot((yA+1)/2.) #label propagation
        if len(yB):
            yestim = np.argmax(Yst,1)
            ytruth = np.argmax(yB,1)
            #print(list(yestim-ytruth).count(0)/nB)
        #Yst=ntest*G.T.dot(y_f)
        g.fit(Kt,Yst)
        ypred=g.predict(Kt)

        
        Chinge=classif.loss_hinge(yA,ypred)
        #Chinge=SVMclassifier.loss_hinge(y_f*2-1,ypred*2-1)
        
        #C=alpha*C0+Chinge

        if len(yB):
            TBR1=np.mean(yB==np.argmax(ypred,1))
            TBR.append(TBR1)
            

    results['ypred']=np.argmax(ypred,1)
    if len(yB):
        results['TBR']=TBR

    results['clf']=g
    results['Gs']=Gs
    results['Gv']=Gv
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost
    
    return g,results




# model is a nn compiled with l2 loss
# YA is (nA,nclass) --> one-hot-encoded labels of XA
# yB is (nB,) --> labels of XB
# XBtest,yBtest : test target data (only to evaluate the model with unseen data over iterations)

def jdcot_nn_l2(model,XA,YA,XB,yB=[],XBtest=[],yBtest=[],
                wA=None,wB=None,vA=None,vB=None,random_init=False,
                algo='emd', reg=0,algo2='emd',reg2=0,alpha=1,
                numIterBCD = 10, nb_epoch=10,batch_size=10):
    """

    Args:
      model:
      XA:
      YA:
      XB:
      yB:
      XBtest:
      yBtest:
      wA:
      wB:
      vA:
      vB:
      random_init:
      algo:
      reg:
      algo2:
      reg2:
      alpha:
      numIterBCD:
      nb_epoch:
      batch_size:

    Returns:

    """

    # Initializations
    nA,dA = XA.shape
    nB,dB = XB.shape

    LPS = []    # label propagation score
    train_score = []
    test_score = []

    if vA is None:
        vA = np.ones(dA) / dA  # is (d,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (d',)
    if wA is None:
        wA = np.ones(nA) / nA  # is (n,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (n',)

    if not random_init:
        Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
        Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
    else:
        Gs=random_gamma_init(wA,wB)
        Gv=random_gamma_init(vA,vB)

    # original loss
    C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)
    cost = np.inf

    Gsold = Gs
    Gvold = Gv
    costold = cost


    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    # function cost initialization
    fcost = np.zeros((nA,nB))  # is (nA,nB)

    # BCD iterations : laternate samples and variables mappings optimisations
    for k in range(numIterBCD):
        #print('num iter BCD:',k+1)

        # step 1 : samples coupling optimisation
        Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +fcost # is (nA,nB)
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : variables coupling optimisation
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA,vB, Mv, reg2)

        # label propagation : estimated labels of XB using transport map on samples from A
        YBhat=nB*Gs.T.dot(YA)        # soft labels - is (nB,nclass)
        yestim = np.argmax(YBhat,1)  # classification decision --> hard labels - is (nB,)

        # training model on target data with estimated labels
        model.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
        ypred=model.predict(XB)
        if len(XBtest):
         yval=model.predict(XBtest)
         yval = np.argmax(yval,1)

        # recording scores
        if len(yB):
         LPS.append(list(yestim-yB).count(0)/nB*100)
         train_score.append(list(np.argmax(ypred,1)-yB).count(0)/nB*100)
         #print('label propagation score : ',LPS[-1])
         #print('train score : ',train_score[-1])
        if len(yBtest):
         test_score.append(list(yval-yBtest).count(0)/len(yBtest))
         #print('test score :',list(yval-YBtest).count(0)/XBtest.shape[0])

        #pl.figure()
        #pl.imshow(fcost)
        #pl.show()

        if k>1:
            sav_fcost.append(np.sum(Gs*fcost))
            sav_totalcost.append(np.sum(Gs*Ms))

        # function cost update
        fcost = ot.dist(YA,ypred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)

    if len(yB):
     results['ypred0']=ypred
     results['ypred']=np.argmax(ypred,1)
     results['labProp'] = LPS
     results['train'] = train_score
    if len(yBtest):
        results['test'] = test_score
    results['clf']=model
    results['fcost']=sav_fcost
    results['totalcost']=sav_totalcost


    return model,results





"""
models should be compiled with l2 loss
semisupervison :
YA is (nA,nclass) : one-hot-encoded labels of XA : 1 when the label is known, 0 otherwise
YB is (nB,nclass) : one-hot-encoded labels of XB: 1 when the label is known, 0 otherwise
yAtruth,yBtruth : are the true labels of XA and XB (only used to compute the accuracy, never for learning)
shapeA, shapeB : 2D data shapes (if data are images)
reshape_data : if models are CNN (default=False) 
"""

def jdcot_multitask_classif(modelA,modelB,XA,YA,XB,YB,yAtruth,yBtruth,XAtest=[],yAtest=[],XBtest=[],yBtest=[],
                    wA=None,wB=None,vA=None,vB=None,random_init=False,algo='emd', reg=0,algo2='emd',reg2=0,alpha=1,
                    numIterBCD = 10,nb_epoch=10,batch_size=10,reshape_data=False,shapeA=None,shapeB=None):
    
    idxA = np.where(YA==1)[0]
    idxB = np.where(YB==1)[0]
    
    if len(idxB) == 0 : 
        type_supervision="Unsupervised"
    else : 
        if len(idxA) == XA.shape[0] : 
            type_supervision="Semi"
        else : type_supervision="Partial"
        
    
    
    if type_supervision=="Partial" : 
        # Initializations
        print("Processing Partial JDCOOT method")
        nA,dA = XA.shape
        nB,dB = XB.shape

        if vA is None:
            vA = np.ones(dA) / dA  # is (d,)
        if vB is None:
            vB = np.ones(dB) / dB  # is (d',)
        if wA is None:
            wA = np.ones(nA) / nA  # is (n,)
        if wB is None:
            wB = np.ones(nB) / nB  # is (n',)

        # original losses
        C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
        C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)

        if reshape_data :
         XA = XA.reshape(XA.shape[0],shapeA[0],shapeA[1],1)
         XB = XB.reshape(XB.shape[0],shapeB[0],shapeB[1],1)
         if len(XAtest):
          XAtest = XAtest.reshape(XAtest.shape[0],shapeA[0],shapeA[1],1)
         if len(XBtest):
          XBtest = XBtest.reshape(XBtest.shape[0],shapeB[0],shapeB[1],1)

        # lists of accuracy scores
        trainA = []  
        trainB = []
        testA = []  
        testB = []



        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs=random_gamma_init(wA,wB) 
            Gv=random_gamma_init(vA,vB)

        # models initialization
        #print('initializing models')
        idxA = np.where(YA==1)[0] # indices of examples of XA with known labels
        modelA.fit(XA[idxA],YA[idxA],batch_size=10,epochs=nb_epoch,verbose=0)  # we train the classifier with labelled examples only
        YApred = modelA.predict(XA)  # first estimate of XA labels
        perf = 100*np.mean(np.argmax(YApred,1)==yAtruth)
        trainA.append(perf)
        #print('pred YA init ',perf)
        YApred[idxA]=YA[idxA]    # injection of known labels in the classifier predictions
        if len(XAtest):
         YAtest = modelA.predict(XAtest)  
         yAestim = np.argmax(YAtest,1)
         perfA = 100*np.mean(yAestim==yAtest)
         testA.append(perfA)

        idxB = np.where(YB==1)[0] # indices of examples of XB with known labels
        modelB.fit(XB[idxB],YB[idxB],batch_size=10,epochs=nb_epoch,verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)   # first estimate of XB labels
        perf = 100*np.mean(np.argmax(YBpred,1)==yBtruth)
        trainB.append(perf)
        #print('pred YB init ',perf)
        YBpred[idxB]=YB[idxB]        # injection of known labels in the classifier predictions
        if len(XBtest):
         YBtest = modelB.predict(XBtest)  
         yBestim = np.argmax(YBtest,1)
         perfB = 100*np.mean(yBestim==yBtest)
         testB.append(perfB)


        results = {}


        #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost=loss_crossentropy2(YApred,YBpred)
        cost = []

        for k in range(numIterBCD):
            #print('\n num iter BCD:',k+1) 

            # step 1 : samples coupling optimization 
            Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +fcost # is (nA,nB) 
            if algo == 'emd':
                Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
            elif algo == 'sinkhorn':
                Gs = ot.sinkhorn(wA, wB, Ms, reg)

            # step 2 : features coupling optimization     
            Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
            if algo2 == 'emd':
                Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
            elif algo2 == 'sinkhorn':
                Gv = ot.sinkhorn(vA,vB, Mv, reg2)


            # estimated labels of XA using transport map on samples
            YAhat=nA*Gs.dot(YBpred)
            YAhat[idxA]=YA[idxA]

            # update label estimate with models predictions
            modelA.fit(XA,YAhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
            YApred = modelA.predict(XA)  

            
            YApred[idxA]=YA[idxA]


            # recording source accuracy scores
            yAestim = np.argmax(YApred,1)
            perfA = 100*np.mean(yAestim==yAtruth)
            trainA.append(perfA)
            if len(XAtest):
             YAtest = modelA.predict(XAtest)  
             yAestim = np.argmax(YAtest,1)
             perfA = 100*np.mean(yAestim==yAtest)
             testA.append(perfA)


            YBhat=nB*Gs.T.dot(YApred)
            YBhat[idxB]=YB[idxB]
            modelB.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
            YBpred=modelB.predict(XB)

            #########
            YBpred[idxB]=YB[idxB] 


            # recording target accuracy scores
            yBestim = np.argmax(YBpred,1)
            perfB = 100*np.mean(yBestim==yBtruth)
            trainB.append(perfB)
            if len(XBtest):
             YBtest = modelB.predict(XBtest)  
             yBestim = np.argmax(YBtest,1)
             perfB = 100*np.mean(yBestim==yBtest)
             testB.append(perfB)

            #print('YA pred acc :',perfA)
            #print('YB pred acc :',perfB)

            #pl.figure()
            #pl.imshow(fcost)
            #pl.show()


            # function cost update
            #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost=loss_crossentropy2(YApred,YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
         results['YA test acc'] = testA
        if len(XBtest):
         results['YB test acc'] = testB
        return modelA,modelB,results
    
    if type_supervision=="Semi" : 
        print("Processing Semi Supervsed JDCOOT method")
        # Initializations
        nA,dA = XA.shape
        nB,dB = XB.shape

        if vA is None:
            vA = np.ones(dA) / dA  # is (d,)
        if vB is None:
            vB = np.ones(dB) / dB  # is (d',)
        if wA is None:
            wA = np.ones(nA) / nA  # is (n,)
        if wB is None:
            wB = np.ones(nB) / nB  # is (n',)

        # original losses
        C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
        C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)

        if reshape_data :
         XA = XA.reshape(XA.shape[0],shapeA[0],shapeA[1],1)
         XB = XB.reshape(XB.shape[0],shapeB[0],shapeB[1],1)
         if len(XAtest):
          XAtest = XAtest.reshape(XAtest.shape[0],shapeA[0],shapeA[1],1)
         if len(XBtest):
          XBtest = XBtest.reshape(XBtest.shape[0],shapeB[0],shapeB[1],1)

        # lists of accuracy scores
        trainA = []  
        trainB = []
        testA = []  
        testB = []



        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs=random_gamma_init(wA,wB) 
            Gv=random_gamma_init(vA,vB)

            
        YApred=YA    # injection of known labels in the classifier predictions


        idxB = np.where(YB==1)[0] # indices of examples of XB with known labels
        modelB.fit(XB[idxB],YB[idxB],batch_size=10,epochs=nb_epoch,verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)   # first estimate of XB labels
        


        results = {}


        #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost=loss_crossentropy2(YApred,YBpred)
        cost = []

        for k in range(numIterBCD):
            #print('\n num iter BCD:',k+1) 

            # step 1 : samples coupling optimization 
            Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + fcost # is (nA,nB) 

            if algo == 'emd':
                Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
            elif algo == 'sinkhorn':
                Gs = ot.sinkhorn(wA, wB, Ms, reg)

            # step 2 : features coupling optimization     
            Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
            if algo2 == 'emd':
                Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
            elif algo2 == 'sinkhorn':
                Gv = ot.sinkhorn(vA,vB, Mv, reg2)


            YApred=YA



            YBhat=nB*Gs.T.dot(YApred)
            YBhat[idxB]=YB[idxB]
            modelB.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
            YBpred=modelB.predict(XB)

            #########
            YBpred[idxB]=YB[idxB] 


            # recording target accuracy scores
            yBestim = np.argmax(YBpred,1)
            perfB = 100*np.mean(yBestim==yBtruth)
            trainB.append(perfB)
            if len(XBtest):
             YBtest = modelB.predict(XBtest)  
             yBestim = np.argmax(YBtest,1)
             perfB = 100*np.mean(yBestim==yBtest)
             testB.append(perfB)

            #print('YA pred acc :',perfA)
            #print('YB pred acc :',perfB)

            #pl.figure()
            #pl.imshow(fcost)
            #pl.show()


            # function cost update
            #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost=loss_crossentropy2(YApred,YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
         results['YA test acc'] = testA
        if len(XBtest):
         results['YB test acc'] = testB
        return modelA,modelB,results
    
    
    
    if type_supervision=="Unsupervised" : 
        print("Processing Unsupervised JDCOOT method")
        
                # Initializations
        nA,dA = XA.shape
        nB,dB = XB.shape

        if vA is None:
            vA = np.ones(dA) / dA  # is (d,)
        if vB is None:
            vB = np.ones(dB) / dB  # is (d',)
        if wA is None:
            wA = np.ones(nA) / nA  # is (n,)
        if wB is None:
            wB = np.ones(nB) / nB  # is (n',)

        # original losses
        C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
        C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)

        if reshape_data :
         XA = XA.reshape(XA.shape[0],shapeA[0],shapeA[1],1)
         XB = XB.reshape(XB.shape[0],shapeB[0],shapeB[1],1)
         if len(XAtest):
          XAtest = XAtest.reshape(XAtest.shape[0],shapeA[0],shapeA[1],1)
         if len(XBtest):
          XBtest = XBtest.reshape(XBtest.shape[0],shapeB[0],shapeB[1],1)

        # lists of accuracy scores
        trainA = []  
        trainB = []
        testA = []  
        testB = []



        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs=random_gamma_init(wA,wB) 
            Gv=random_gamma_init(vA,vB)

            
        YApred=YA    # injection of known labels in the classifier predictions


        
    # step 1 : samples coupling optimization 
        Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) # is (nA,nB)          
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

            # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA,vB, Mv, reg2)


        YApred=YA

        YBhat=nB*Gs.T.dot(YApred)
        modelB.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
        YBpred=modelB.predict(XB)      


        results = {}


        #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost=loss_crossentropy2(YApred,YBpred)
        cost = []

        for k in range(numIterBCD):
            #print('\n num iter BCD:',k+1) 

            # step 1 : samples coupling optimization 
            Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +fcost # is (nA,nB)          
            if algo == 'emd':
                Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
            elif algo == 'sinkhorn':
                Gs = ot.sinkhorn(wA, wB, Ms, reg)

            # step 2 : features coupling optimization     
            Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
            if algo2 == 'emd':
                Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
            elif algo2 == 'sinkhorn':
                Gv = ot.sinkhorn(vA,vB, Mv, reg2)


            YApred=YA



            YBhat=nB*Gs.T.dot(YApred)
            modelB.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
            YBpred=modelB.predict(XB)



            #print('YA pred acc :',perfA)
            #print('YB pred acc :',perfB)

            #pl.figure()
            #pl.imshow(fcost)
            #pl.show()


            # function cost update
            #fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost=loss_crossentropy2(YApred,YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
         results['YA test acc'] = testA
        if len(XBtest):
         results['YB test acc'] = testB
        return modelA,modelB,results




"""
jdcot multi-task for multi regression problems
npreds : number of parameters to predict (it has to be the same number for both datasets)
yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
YA is (nA,npreds), YB is (nB,npreds) : line of NaN if non observed labels and true values if observed labels (semi supervision)
"""

def jdcot_multitask_reg(modelA,modelB,XA,YA,XB,YB,yAtruth,yBtruth,XAtest=[],XBtest=[],yAtest=[],yBtest=[],
                    shapeA=None,shapeB=None,wA=None,wB=None,vA=None,vB=None,random_init=False,
                    algo='emd', reg=0,algo2='emd',reg2=0,alpha=1,
                    numIterBCD = 10,nb_epoch=10,batch_size=10,reshape_data=True):
    
    trainA = []  
    trainB = []
    testA = []  
    testB = []
    results = {}
    
    # Initializations
    nA,dA = XA.shape
    nB,dB = XB.shape
    
    ######
    
    shapeA=[dA,1]#1 can be replaced by the number of variables of Y to pred
    shapeB=[dB,1]
    #####
    
    if vA is None:
        vA = np.ones(dA) / dA  # is (d,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (d',)
    if wA is None:
        wA = np.ones(nA) / nA  # is (n,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (n',)
    
     # original losses
    C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)
    
    if not random_init:
        Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
        Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
    else:
        Gs=random_gamma_init(wA,wB) 
        Gv=random_gamma_init(vA,vB)
        
    if reshape_data :
        XA = XA.reshape(XA.shape[0],shapeA[0],shapeA[1],1)
        XB = XB.reshape(XB.shape[0],shapeB[0],shapeB[1],1)
        #XAtest = XAtest.reshape(XAtest.shape[0],shapeA[0],shapeA[1],1)
        #XBtest = XBtest.reshape(XBtest.shape[0],shapeB[0],shapeB[1],1)

    # models initialization
    #print('initializing models')
    idxA = np.where(~np.isnan(YA))[0] # indices of examples of XA with known labels
    
    modelA.fit(XA[idxA],YA[idxA],batch_size=10,epochs=nb_epoch,verbose=0)  # we train the model with labelled examples only
    YApred = modelA.predict(XA)  # first estimate of XA labels
    perf = modelA.evaluate(XA,yAtruth)
    trainA.append(perf)
    #print('pred YA init ',perf)
    #print(YApred.shape)
    #print(YA)
    #print(idxA)
    YApred[idxA]=YA[idxA]    # injection of known labels in the model predictions
    
    idxB = np.where(~np.isnan(YB))[0]# indices of examples of XB with known labels
    
    if len(idxB) == 0 : #first initialisation in case of unsupervised treatment (COOT on YB)
        
        M_lin = None

        Ts, Tv, cost = cot_numpy(X1=XA, 
                                    X2=XB, 
                                         niter=100, C_lin=M_lin,
                                                     algo='sinkhorn',reg=1,
                                                     algo2='emd', verbose = False)

        # Target estimation
        #PLUS DE OH ENC
        YBpred = nB*np.dot(Ts.T,YApred).reshape((-1,1))
        #print(YBpred)
            
        
    else : 
        modelB.fit(XB[idxB],YB[idxB],batch_size=10,epochs=nb_epoch,verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)    # first estimate of XB labels
        perf = modelB.evaluate(XB,yBtruth)
        trainB.append(perf)
        #print('pred YB init ',perf)
        YBpred[idxB]=YB[idxB]        # injection of known labels in the model predictions
    
    #print(YApred)
    #print(YBpred)
    fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
    #fcost = np.zeros((nA,nB))
    
    cost = []
    
    for k in range(numIterBCD):
        #print('\n num iter BCD:',k+1) 
        
        # step 1 : samples coupling optimization 
        Ms = alpha*(C_s - np.dot(h1_s, Gv).dot(h2_s.T)) +fcost # is (nA,nB)          
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)                # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA,vB, Mv, reg2)
        

        # estimated labels of XA using transport map on samples
        YAhat=nA*Gs.dot(YBpred)
        YAhat[idxA]=YA[idxA]
        
        # update label estimate with models predictions
        modelA.fit(XA,YAhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
        YApred = modelA.predict(XA) 
        YApred[idxA]=YA[idxA]
        perfA = modelA.evaluate(XA,yAtruth)
        trainA.append(perfA)
        

        YBhat=nB*Gs.T.dot(YApred)
        YBhat[idxB]=YB[idxB]
        modelB.fit(XB,YBhat,batch_size=batch_size,epochs=nb_epoch,verbose=0)
        YBpred=modelB.predict(XB)
        YBpred[idxB]=YB[idxB] 
        perfB = modelB.evaluate(XB,yBtruth)
        trainB.append(perfB)
 
        #print('YA pred acc :',perfA)
        #print('YB pred acc :',perfB)
        
        #pl.figure()
        #pl.imshow(fcost)
        #pl.show()

            
        # function cost
        fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)


    results['YA train acc'] = trainA
    results['YB train acc'] = trainB
    #results['YA test acc'] = testA
    #results['YB test acc'] = testB
    return modelA,modelB,results


# @title CLASSIF

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 16:54:30 2017

@author: rflamary
"""

# Author: Remi Flamary <remi.flamary@unice.fr>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License


#import numpy as np
#import sklearn
#import scipy.optimize as spo
#from sklearn.model_selection import KFold
#from scipy.spatial.distance import cdist
#from sklearn.metrics.pairwise import rbf_kernel

#import time
__time_tic_toc=time.time()


def get_label_matrix(y):
    vals=np.unique(y)

    # class matrices for source
    Y=np.zeros((len(y),len(vals)))
    Yb=np.zeros((len(y),len(vals)))
    for i,val in enumerate(vals):
        Y[:,i]=2*((y==val)-.5)
        Yb[:,i]=(y==val)
    return Y,Yb

def estimGamma(X):
    return 1./(2*(np.median(cdist(X,X,'euclidean'))**2))


def tic():
    global __time_tic_toc
    __time_tic_toc=time.time()

def toc(message='Elapsed time : {} s'):
    t=time.time()
    print(message.format(t-__time_tic_toc))
    return t-__time_tic_toc

def toq():
    t=time.time()
    return t-__time_tic_toc

def loss_hinge(Y,F):
    res=np.zeros((Y.shape[0],F.shape[0]))
    for i in range(Y.shape[1]):
        res+=np.maximum(0,1-Y[:,i].reshape((Y.shape[0],1))*F[:,i].reshape((1,F.shape[0])))**2
    return res

class Classifier:
    # cross validate parameters with k-fold classification


    def crossval(self,X,Y,kerneltype='linear',nbsplits=5,g_range=np.logspace(-3,3,7),l_range = np.logspace(-3,0,4)):
        kf = KFold(n_splits=nbsplits)
        if kerneltype=='rbf':
            dim=(len(g_range),len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)

            for i,g in enumerate(g_range):
                for j,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.rbf_kernel(X[train,:],gamma=g)
                        Kt=sklearn.metrics.pairwise.rbf_kernel(X[train,:],X[test,:],gamma=g)
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)

                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)

                        results[i,j] += np.mean(ydec==yt)
            results = results /nbsplits
            #print results

            i,j = np.unravel_index(results.argmax(), dim)

            self.lambd=l_range[j]

            return g_range[i],l_range[j]
        else:
            dim=(len(l_range))
            results = np.zeros(dim)
            kf = KFold(n_splits=nbsplits, shuffle=True)
            for i,l in enumerate(l_range):
                    self.lambd=l
                    for train, test in kf.split(X):
                        K=sklearn.metrics.pairwise.linear_kernel(X[train,:])
                        Kt=sklearn.metrics.pairwise.linear_kernel(X[train,:],X[test,:])
                        self.fit(K,Y[train,:])
                        ypred=self.predict(Kt.T)
                        ydec=np.argmax(ypred,1)
                        yt=np.argmax(Y[test,:],1)
                        results[i] += np.mean(ydec==yt)
            results = results /nbsplits


            self.lambd=l_range[results.argmax()]

            return self.lambd


def hinge_squared_reg(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[0],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()

def hinge_squared_reg_bias(w,X,Y,lambd):
    """
    compute loss dans gradient for squared hing loss with quadratic regularization

    """
    nbclass=Y.shape[1]
    w=w.reshape((X.shape[1],Y.shape[1]))
    f=X.dot(w)

    err_alpha=np.maximum(0,1-f)
    err_alpha1=np.maximum(0,1+f)

    loss=0
    grad=np.zeros_like(w)
    for i in range(nbclass):
        loss+=Y[:,i].T.dot(err_alpha[:,i]**2)+(1-Y[:,i]).T.dot(err_alpha1[:,i]**2)
        grad[:,i]+=2*X.T.dot(-Y[:,i]*err_alpha[:,i]+(1-Y[:,i])*err_alpha1[:,i]) # alpha

    # regularization term
    w[:,-1]=0
    loss+=lambd*np.sum(w**2)/2
    grad+=lambd*w

    return loss,grad.ravel()



class SVMClassifier(Classifier):

    def __init__(self,lambd=1e-2,bias=False):
        self.lambd=lambd
        self.w=None
        self.bias=bias

        # w = argmin de la loss, f = val min de la loss


    def fit(self,K,y):
        # beware Y is a binary matrix to allow for more general solvers (see JDOT)
        if self.bias:
            K1=np.hstack((K,np.ones((K.shape[0],1))))
            self.w=np.zeros((K1.shape[1],y.shape[1]))
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg_bias(w,X=K1,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)
            self.b=self.w.reshape((K1.shape[1],y.shape[1]))[-1,:]
            self.w=self.w.reshape((K1.shape[1],y.shape[1]))[:-1,:]

        else:
            self.w=np.zeros((K.shape[1],y.shape[1])) # K.shape[1] = n ou n' , y.shape[1] = nb classes
            self.w,self.f,self.log=spo.fmin_l_bfgs_b(lambda w: hinge_squared_reg(w,X=K,Y=y,lambd=self.lambd),self.w,maxiter=1000,maxfun=1000)
            self.w=self.w.reshape((K.shape[1],y.shape[1]))

    def predict(self,K):
        if self.bias:
            return np.dot(K,self.w)+self.b
        else:
            return np.dot(K,self.w)



class KRRClassifier(Classifier):

    def __init__(self,lambd=1e-2):
        self.lambd=lambd

    def fit(self,K,y,sw=False):
        ns=K.shape[0]
        if sw:
            K=K*sw
        K0=np.vstack((np.hstack((np.eye(ns),np.zeros((ns,1)))),np.zeros((1,ns+1))))

        ## true reg in RKHS
        #K0=np.vstack((np.hstack((K,np.zeros((ns,1)))),np.zeros((1,ns+1))))

        K1=np.hstack((K,np.ones((ns,1))))
        if sw:
            y1=K1.T.dot(y*sw)
        else:
            y1=K1.T.dot(y)

        temp=np.linalg.solve(K1.T.dot(K1) + self.lambd*K0,y1)
        self.w,self.b=temp[:-1],temp[-1]

    def predict(self,K):
        return np.dot(K,self.w)+self.b


# -

# # Data Generation

# +
"""
data_generator :  Simulate Data from a Scenario with chosen parameters

Input : n_Source, n_Target (int): number of observation of source/target

        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        
        Cols_Chosen_For_Observation_Source (Array of str :  ['X1', ... , 'Xd']) : Array of names of observed variables in source (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly 
        
        Cols_Chosen_For_Observation_Target, :['X1', ... , 'Xd']) : Array of names of observed variables in target (In order to keep the same observations when we want to compare performance with test sample) If None, chosen randomly 
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes

Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis, 
                                         Z the discrete objective variable for classification analysis

"""

def data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                   mean_Y_Target,Source_Generation_Correlation, Source_Non_Generation_Correlation,
                   Target_Generation_Correlation, Target_Non_Generation_Correlation,
                   Sparse_Rate,
                   Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target,
                   Observed_Covariates_Proportion_Source,
                   Observed_Covariates_Proportion_Target,
                   Indexes_Chosen_For_Generation=None,Cols_Chosen_For_Observation_Source=None,
                   Cols_Chosen_For_Observation_Target=None,Poisson=False):
    # General parameters
    
    if any(Indexes_Chosen_For_Generation)==None:
        Indexes_Chosen_For_Generation = np.random.choice(np.arange(d_Source), math.ceil(Sparse_Rate * d_Source),
                                                     replace=False)
        #same variables to generate target and source

    # Source covariables generation
     
    cov_Source=np.eye(d_Source)
    cov_Target=np.eye(d_Target)
    
    I=0
    
    for i in Indexes_Chosen_For_Generation :
        J=0
        for j in Indexes_Chosen_For_Generation :
            cov_Source[i,j]=Source_Generation_Correlation**abs(I-J)
            J=J+1
        I=I+1
        
    
    I=0
    
    for i in np.setdiff1d(np.arange(d_Source),Indexes_Chosen_For_Generation) :
        J=0
        for j in np.setdiff1d(np.arange(d_Source),Indexes_Chosen_For_Generation)  :
            cov_Source[i,j]=Source_Non_Generation_Correlation**abs(I-J)
            J=J+1
        I=I+1
    
    
    
    
    #cov_Source[Indexes_Chosen_For_Generation,Indexes_Chosen_For_Generation]=Source_Generation_Correlation
    
    
    I=0
    for i in Indexes_Chosen_For_Generation :
        J=0
        for j in Indexes_Chosen_For_Generation :
            cov_Target[i,j]=Target_Generation_Correlation**abs(I-J)
            J=J+1
        I=I+1
        
    
    #MEME NOMBRE DE VARIABLES DANS SOURCE ET DANS TARGET CAR MEME  MONDE
    I=0
    
    for i in np.setdiff1d(np.arange(d_Source),Indexes_Chosen_For_Generation) :
        J=0
        for j in np.setdiff1d(np.arange(d_Source),Indexes_Chosen_For_Generation)  :
            cov_Target[i,j]=Target_Non_Generation_Correlation**abs(I-J)
            J=J+1
        I=I+1
    
    #print(cov_Source[np.ix_(Indexes_Chosen_For_Generation,Indexes_Chosen_For_Generation)])
    
    #print(cov_Source[np.ix_(np.setdiff1d(np.arange(0, d_Source - 1),Indexes_Chosen_For_Generation) ,np.setdiff1d(np.arange(0, d_Source - 1),Indexes_Chosen_For_Generation) )])

    
    
    X_Source = np.random.multivariate_normal(mean_X_Source, cov_Source, n_Source)

    # Source objective variables generation

    # Continuous
    # Parameters of regression
    # Coefficients for linear predictor

    if sum(mean_X_Source[Indexes_Chosen_For_Generation]) == 0 or mean_Y_Source == 0:
        b_source = 1
    else:
        b_source = mean_Y_Source / sum(mean_X_Source[Indexes_Chosen_For_Generation])

    a_Source = np.zeros(d_Source)
    a_Source[Indexes_Chosen_For_Generation] = b_source

    # Sigma
    sigma_Source = np.var(np.dot(X_Source, a_Source)) * (1 - R2_Source) / R2_Source

    # Variable generation
    Y_Source = np.dot(X_Source, a_Source) + np.random.normal(loc=0, scale=np.sqrt(sigma_Source), size=n_Source)

    # Discrete
    # Parameters of regression
    # Coefficients for linear predictor
    a_Source = np.zeros(d_Source)
    a_Source[Indexes_Chosen_For_Generation] = np.log(Odds_Ratio_Source)

    # Variable generation
    proba_Z_Source = np.exp(np.dot(X_Source, a_Source)) / (1 + np.exp(np.dot(X_Source, a_Source)))

    US = np.random.uniform(0, 1, n_Source)

    Z_Source = np.zeros(n_Source)
    Z_Source[US < proba_Z_Source] = 1
    
    if Poisson :
        at = np.zeros(d_Source)
        at[Indexes_Chosen_For_Generation] = 10/100
        Z_Source=np.random.poisson(np.exp(np.dot(X_Source, at)),n_Source)

    # Target covariables generation
    X_Target = np.random.multivariate_normal(mean_X_Target, cov_Target, n_Target)
    # Target objective variables generation
    # Continuous
    # Parameters of regression
    # Coefficients for linear predictor

    if sum(mean_X_Target[Indexes_Chosen_For_Generation]) == 0 or mean_Y_Target == 0:
        b_Target = 1
    else:
        b_Target = mean_Y_Target / sum(mean_X_Target[Indexes_Chosen_For_Generation])

    a_Target = np.zeros(d_Target)
    a_Target[Indexes_Chosen_For_Generation] = b_Target

    # Sigma
    sigma_Target = np.var(np.dot(X_Target, a_Target)) * (1 - R2_Target) / R2_Target

    # Variable generation
    Y_Target = np.dot(X_Target, a_Target) + np.random.normal(loc=0, scale=np.sqrt(sigma_Target), size=n_Target)

    # Discrete
    # Parameters of regression
    # Coefficients for linear predictor
    a_Target = np.zeros(d_Target)
    a_Target[Indexes_Chosen_For_Generation] = np.log(Odds_Ratio_Target)

    # Variable generation
    proba_Z_Target = np.exp(np.dot(X_Target, a_Target)) / (1 + np.exp(np.dot(X_Target, a_Target)))

    US = np.random.uniform(0, 1, n_Target)

    Z_Target = np.zeros(n_Target)
    Z_Target[US < proba_Z_Target] = 1
    
    if Poisson :
        at = np.zeros(d_Target)
        at[Indexes_Chosen_For_Generation] = 10/100
        Z_Target=np.random.poisson(np.exp(np.dot(X_Target, at)),n_Target)
        
    
    
    if Cols_Chosen_For_Observation_Source is None or Cols_Chosen_For_Observation_Target is None :

           
        
        # Source covariables mask (prop% in covariables used for generation + prop% in covariables not used for generation)
        Observed_Covariables_Indexes_Source1 = np.random.choice(Indexes_Chosen_For_Generation,
                                                               math.floor(Observed_Covariates_Proportion_Source * len(Indexes_Chosen_For_Generation)),
                                                               replace=False)
        
        
        Observed_Covariables_Indexes_Source2 = np.random.choice(np.setdiff1d(np.arange(d_Source),Indexes_Chosen_For_Generation),
                                                                math.floor(Observed_Covariates_Proportion_Source * (d_Source-len(Indexes_Chosen_For_Generation))),
                                                                replace=False)
        
        X_source_masked = X_Source[:, np.union1d(Observed_Covariables_Indexes_Source1,Observed_Covariables_Indexes_Source2)]



        # Target covariables mask (prop% in covariables used for generation + prop% in covariables not used for generation)
        Observed_Covariables_Indexes_Target1 = np.random.choice(Indexes_Chosen_For_Generation,
                                                                math.floor(Observed_Covariates_Proportion_Target * len(Indexes_Chosen_For_Generation)),
                                                                replace=False)
        
        Observed_Covariables_Indexes_Target2 = np.random.choice(
            np.setdiff1d(np.arange(d_Target), Indexes_Chosen_For_Generation),
            math.floor(Observed_Covariates_Proportion_Target * (d_Target - len(Indexes_Chosen_For_Generation))),
            replace=False)
        
        X_target_masked = X_Target[:,np.union1d(Observed_Covariables_Indexes_Target1, Observed_Covariables_Indexes_Target2)]

    

        

        # Datasets
        #Source
        data_Source = pd.DataFrame(np.c_[X_source_masked, Y_Source, Z_Source])
        col_names = ['X' + str(i) for i in np.union1d(Observed_Covariables_Indexes_Source1,Observed_Covariables_Indexes_Source2)+1] + ['Y', 'Z']
        data_Source.columns = col_names

        #Target
        data_Target = pd.DataFrame(np.c_[X_target_masked, Y_Target, Z_Target])
        col_names = ['X' + str(i) for i in np.union1d(Observed_Covariables_Indexes_Target1, Observed_Covariables_Indexes_Target2)+1] + ['Y', 'Z']
        data_Target.columns = col_names
    
    
    else : 
        X_source_masked = X_Source
        X_target_masked = X_Target
        
        # Datasets
        #Source
        data_Source = pd.DataFrame(np.c_[X_source_masked, Y_Source, Z_Source])
        col_names = ['X' + str(i) for i in np.arange(1,d_Source+1)] + ['Y', 'Z']
        data_Source.columns = col_names
       
        

        #Target
        data_Target = pd.DataFrame(np.c_[X_target_masked, Y_Target, Z_Target])
        col_names = ['X' + str(i) for i in np.arange(1,d_Target+1)] + ['Y', 'Z']
        data_Target.columns = col_names
        
        data_Source = data_Source.loc[:,list(Cols_Chosen_For_Observation_Source)+ ['Y', 'Z']]
        data_Target = data_Target.loc[:,list(Cols_Chosen_For_Observation_Target)+ ['Y', 'Z']]
    

    
    return (data_Source, data_Target)


# -

# ## Reference Scenario

# +
"""
Sref : Sample a reference scenario as described in the paper

Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 

Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis, 
                                         Z the discrete objective variable for classification analysis
"""

def Sref(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    n_S = 1000
    n_T = 1000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75

    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    #cov_S = np.eye(d_S)
    #cov_T = np.eye(d_T)
    rho_source_generation=0.7
    rho_source_non_generation=0.2
    rho_target_generation=0.7
    rho_target_non_generation=0.2
    
    return(data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                         pxo_S, pxo_T,Indexes_Chosen_For_Generation))

"""
Sref_test : Sample a test sample for the reference scenario (300 observations)

Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 

Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis, 
                                         Z the discrete objective variable for classification analysis
"""
def Sref_test(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    n_S = 300
    n_T = 300
    d = 100
    d_S = d
    d_T = d
    pxo = 1
    sr=0.75
    pxo_S = pxo
    pxo_T = pxo
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    #cov_S = np.eye(d_S)
    #cov_T = np.eye(d_T)
    rho_source_generation=0.7
    rho_source_non_generation=0.2
    rho_target_generation=0.7
    rho_target_non_generation=0.2
    
    return(data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                         pxo_S, pxo_T,Indexes_Chosen_For_Generation))



# -

# ## Reference scenario in case of multiclassification

# +
"""
Sref_Poisson : Sample a reference scenario but with more than 2 classes for the classification analysis

Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 

Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis, 
                                         Z the discrete objective variable for classification analysis
"""


def Sref_Poisson(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    n_S = 10000
    n_T = 10000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    #cov_S = np.eye(d_S)
    #cov_T = np.eye(d_T)
    rho_source_generation=0.7
    rho_source_non_generation=0.2
    rho_target_generation=0.7
    rho_target_non_generation=0.2
    
    return(data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                         pxo_S, pxo_T,Indexes_Chosen_For_Generation,Poisson=True))

"""
Sref_Poisson_test : Sample a test sample for the Poisson reference scenario (3000 observations)

Input : Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 

Output : Data (tuple): tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis, 
                                         Z the discrete objective variable for classification analysis
"""
def Sref_Poisson_test(Indexes_Chosen_For_Generation=INDEX_GENERATION):
    n_S = 3000
    n_T = 3000
    d = 100
    d_S = d
    d_T = d
    pxo = 1
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    #cov_S = np.eye(d_S)
    #cov_T = np.eye(d_T)
    rho_source_generation=0.7
    rho_source_non_generation=0.2
    rho_target_generation=0.7
    rho_target_non_generation=0.2
    
    return(data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T, mY_S, mY_T, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, sr, OR_S, OR_T, R2_S, R2_T,
                         pxo_S, pxo_T,Indexes_Chosen_For_Generation,Poisson=True))


# -

# # Performance Measurement

# +
"""
Performance : Compute efficiency of the methods with the given data. Can be applied to any data if the format is respected.

Input : Data (tuple) : Learning data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
        Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            If None, Learning data is splited into 30% of test data and 70% of learning data
                            
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'partial', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and Labelled_Proportion_Source*100 % of the observations of source are labelled
                                                                                                       if 'cross-partial', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and Labelled_Proportion_Source*100 % of the observations of source are labelled. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        

Output : tuple of 6 elements : Pure (transport) performance of COOT,
                               Pure (transport) performance of JDCOOT,
                               Test (trained model applied to test data) performance of COOT,
                               Test (trained model applied to test data) performance of JDCOOT,
                               Test (trained model applied to test data) performance of REFERENCE,
                               Pure (transport) performance of REFERENCE
"""



def Performance(Data, Data_test=None,Objective_Variable='discrete' ,algo='both' ,type_supervision='unsupervised',Balance=True,alpha=None, Labelled_Proportion_Target=None,Labelled_Proportion_Source=None,    ):
    S=Data[0]
    T=Data[1]

    if Data_test==None :
        a=np.random.choice(np.arange(len(S.index)),math.ceil(0.7*len(S.index)),replace=False)
        b=np.random.choice(np.arange(len(T.index)),math.ceil(0.7*len(T.index)),replace=False)
        S_test=S.iloc[a,:].reset_index(drop=True)
        T_test=T.iloc[b,:].reset_index(drop=True)

        S=S.iloc[np.setdiff1d(np.arange(len(S.index)),a),:].reset_index(drop=True)
        T=T.iloc[np.setdiff1d(np.arange(len(T.index)),b),:].reset_index(drop=True)
    else : 
        S_test=Data_test[0]
        S_test=S_test.loc[:,S.columns]
        T_test=Data_test[1]
        T_test=T_test.loc[:,T.columns]
        
        #print(S)
        #print(S_test)
    

    if Balance==True and (Objective_Variable=='discrete' or Objective_Variable=='both'):
        if len(np.unique(S['Z']))>2:
            
            del_idx=np.array([])
            for k in np.unique(S['Z']):
                if sum(S['Z']==k)<0.01*len(S['Z']):
                    del_idx=np.append(del_idx,k)
            S=S.loc[~np.in1d(S['Z'],del_idx),:].reset_index(drop=True)
            
        
        if len(np.unique(T['Z']))>2:
            del_idx=np.array([])
            for k in np.unique(T['Z']):
                if sum(T['Z']==k)<0.01*len(T['Z']):
                    del_idx=np.append(del_idx,k)
            T=T.loc[~np.in1d(T['Z'],del_idx),:].reset_index(drop=True)
            
            
        S_nPerClass =math.ceil(min(np.unique(S['Z'], return_counts=True)[1])) #number of observations kept referenced by the min number of available observation per class
        T_nPerClass = math.ceil(min(np.unique(T['Z'], return_counts=True)[1]))
        
        z_kept_source=np.array([]).astype(int)
        for lab in np.unique(S['Z']) : 
            z_kept_source=np.append(z_kept_source,np.random.choice(np.where(S['Z']==lab)[0],S_nPerClass,replace=False))
        
        S=S.loc[z_kept_source,:].reset_index(drop=True)
        
        z_kept_target=np.array([]).astype(int)
        for lab in np.unique(T['Z']) : 
            z_kept_target=np.append(z_kept_target,np.random.choice(np.where(T['Z']==lab)[0],T_nPerClass,replace=False))
        T=T.loc[z_kept_target,:].reset_index(drop=True)

    #print(np.unique(S['Z'],return_counts=True))
    #print(np.unique(T['Z'],return_counts=True))
    
    if type_supervision=='unsupervised':
        prop_S=1
        prop_T=0
        if alpha==None :
            if Objective_Variable=='discrete':
                alpha=0.661
            else : alpha=0.3
    elif type_supervision=='semi-supervised' and Labelled_Proportion_Target != None:
        prop_S=1
        prop_T=Labelled_Proportion_Target
        if alpha==None :
            if Objective_Variable=='discrete':
                alpha=3.335
            else : alpha=2.625
    elif (type_supervision=='partial' or  type_supervision == 'cross-partial') and Labelled_Proportion_Source != None and Labelled_Proportion_Target != None:
        prop_S=Labelled_Proportion_Source
        prop_T=Labelled_Proportion_Target
        if alpha==None :
            if Objective_Variable=='discrete':
                alpha=2.875
            else : alpha=2.425
    else : return -1
    
    
    
    if Objective_Variable=='discrete' or Objective_Variable=='both' :
        z_labelled_source=np.array([]).astype(int)
        for lab in np.unique(S['Z']) : 
            a=np.random.choice(np.where(S['Z']==lab)[0],math.ceil(prop_S*sum(S['Z']==lab)),replace=False)
            z_labelled_source=np.append(z_labelled_source,a)

        z_labelled_target=np.array([]).astype(int)
        for lab in np.unique(T['Z']) : 
            b=np.random.choice(np.where(T['Z']==lab)[0],math.ceil(prop_T*sum(T['Z']==lab)),replace=False)
            z_labelled_target=np.append(z_labelled_target,b)

          
    
        Z_training_data_source=S.loc[:,S.columns != 'Y']
        Z_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),'Z']=-1

        Z_training_data_Target=T.loc[:,T.columns != 'Y']
        Z_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z']=-1
    
    
    if Objective_Variable=='both' :
        #Memes indexs labellisés
        y_labelled_source=z_labelled_source
        y_labelled_target=z_labelled_target
        Y_training_data_source=S.loc[:, S.columns != 'Z']
        Y_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Y']=np.NaN
        Y_training_data_Target=T.loc[:, T.columns != 'Z']
        Y_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y']=np.NaN
        
    if Objective_Variable=='continuous' : 
        y_labelled_source=np.random.choice(np.arange(len(S['Y'])),math.ceil(prop_S*len(S['Y'])),replace=False)
        y_labelled_target=np.random.choice(np.arange(len(T['Y'])),math.ceil(prop_T*len(T['Y'])),replace=False)
        Y_training_data_source=S.loc[:, S.columns != 'Z']
        Y_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Y']=np.NaN
        Y_training_data_Target=T.loc[:, T.columns != 'Z']
        Y_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y']=np.NaN
        
                          
    perf_coot=np.array([])
    perf_jdcoot=np.array([])
    perf_coot_test=np.array([])
    perf_jdcoot_test=np.array([])
    perf_ref=np.array([])
    perf_ref2=np.array([])
    
    ####reference perf
        
    if type_supervision == 'semi-supervised' and prop_T!=0 and prop_T!=1: 
        
        if Objective_Variable=='discrete' or Objective_Variable=='both' : 
            def clf_seq(shape,nClass):
                        model = tf_keras.Sequential([
                        Dense(units=128,input_shape=shape,activation='sigmoid'),
                        Dense(units=nClass,activation = 'sigmoid')        ])
                        return model

            vfunc = np.vectorize(lambda arr : 'X' in arr)
            fe_size = sum(vfunc(T.columns)) #Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
                    #loss = 'MeanSquaredError
            clf = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
            clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
            enc = onehot(handle_unknown='ignore',sparse=False,categories=[np.arange(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))])
            #print(enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)))

            clf.fit(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,Z_training_data_Target.columns != 'Z'],enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:,vfunc(T.columns)])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            perf_ref=np.append(perf_ref,sum(z_test==T_test.loc[:,'Z'])/len(z_test))
            
            z_test= clf.predict(Z_training_data_Target.loc[Z_training_data_Target['Z']==-1,Z_training_data_Target.columns != 'Z'])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            perf_ref2=np.append(perf_ref2,sum(z_test==T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z'])/len(z_test))
            
        if Objective_Variable=='continuous' or Objective_Variable=='both':
            def clf_seq(shape,nClass):
                        model = tf_keras.Sequential([
                        Dense(units=128,input_shape=shape,activation='linear'),
                        Dense(units=nClass,activation = 'linear')        ])
                        return model

            vfunc = np.vectorize(lambda arr : 'X' in arr)
            fe_size = sum(vfunc(T.columns)) #Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'MeanSquaredError'
            clf = clf_seq(shape, nClass = 1)
            clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])


            clf.fit(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),Y_training_data_Target.columns != 'Y'],Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),'Y'],batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
            #print(len(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),'Y']))
            
            z_test = clf.predict(T_test.loc[:,vfunc(T.columns)]).reshape(-1)
            perf_ref=np.append(perf_ref,sum((z_test-T_test.loc[:,'Y'])**2)/len(z_test))
            
            
            z_test = clf.predict(Y_training_data_Target.loc[np.isnan(Y_training_data_Target['Y']),Y_training_data_Target.columns != 'Y']).reshape(-1)
            perf_ref2=np.append(perf_ref2,sum((z_test-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])**2)/len(z_test))
            #print(perf_ref2)
            
    elif prop_T==0 : 
        if Objective_Variable=='discrete' or Objective_Variable=='both' :
            perf_ref=np.append(perf_ref,np.NaN)
            perf_ref2=np.append(perf_ref2,np.NaN)
        if Objective_Variable=='continuous' or Objective_Variable=='both' :
            perf_ref=np.append(perf_ref,np.NaN)
            perf_ref2=np.append(perf_ref2,np.NaN)
            
    elif type_supervision == 'partial' or  type_supervision == 'cross-partial' :
        
        
        
        if Objective_Variable=='discrete' or Objective_Variable=='both' : 
            def clf_seq(shape,nClass):
                        model = tf_keras.Sequential([
                        Dense(units=128,input_shape=shape,activation='sigmoid'),
                        Dense(units=nClass,activation = 'sigmoid')        ])
                        return model

            vfunc = np.vectorize(lambda arr : 'X' in arr)
            fe_size = sum(vfunc(T.columns)) #Nombre de variables de Target
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
                    #loss = 'MeanSquaredError
            clf = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
            clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
            enc = onehot(handle_unknown='ignore',sparse=False,categories=[np.arange(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))])
            #print(enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)))

            clf.fit(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,Z_training_data_Target.columns != 'Z'],enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:,vfunc(T.columns)])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            
            fe_size = sum(vfunc(S.columns)) #Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
                    #loss = 'MeanSquaredError
            clf2 = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
            
            
            clf2.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
    
            clf2.fit(Z_training_data_source.loc[Z_training_data_source['Z']!=-1,Z_training_data_source.columns != 'Z'],enc.fit_transform(Z_training_data_source.loc[Z_training_data_source['Z']!=-1,'Z'].values.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
            z_test2 = clf2.predict(S_test.loc[:,vfunc(S.columns)])
            z_test2 = enc.inverse_transform(z_test2).reshape(-1)
            
            
            perf_ref=np.append(perf_ref,(sum(z_test==T_test.loc[:,'Z'])+sum(z_test2==S_test.loc[:,'Z']))/(len(z_test)+len(z_test2)))
            
            
            
            
            
            z_test= clf.predict(Z_training_data_Target.loc[Z_training_data_Target['Z']==-1,Z_training_data_Target.columns != 'Z'])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            
            z_test2= clf2.predict(Z_training_data_source.loc[Z_training_data_source['Z']==-1,Z_training_data_source.columns != 'Z'])
            z_test2 = enc.inverse_transform(z_test2).reshape(-1)
            
            perf_ref2=np.append(perf_ref2,(sum(z_test==T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z'])+sum(z_test2==S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),'Z']))/(len(z_test)+len(z_test2)))
            
        if Objective_Variable=='continuous' or Objective_Variable=='both':
            def clf_seq(shape,nClass):
                        model = tf_keras.Sequential([
                        Dense(units=128,input_shape=shape,activation='linear'),
                        Dense(units=nClass,activation = 'linear')        ])
                        return model

            vfunc = np.vectorize(lambda arr : 'X' in arr)
            fe_size = sum(vfunc(T.columns)) #Nombre de variables de Targe
            shape = (fe_size,)
            #loss = 'categorical_crossentropy'
            loss = 'MeanSquaredError'
            clf = clf_seq(shape, nClass = 1)
            clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
            


            clf.fit(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),Y_training_data_Target.columns != 'Y'],Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),'Y'],batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:,vfunc(T.columns)]).reshape(-1)
            
            fe_size = sum(vfunc(S.columns)) #Nombre de variables de Targe
            shape = (fe_size,)
            #loss = 'categorical_crossentropy'
            loss = 'MeanSquaredError'
            clf2 = clf_seq(shape, nClass = 1)
            clf2.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
            
            clf2.fit(Y_training_data_source.loc[~np.isnan(Y_training_data_source['Y']),Y_training_data_source.columns != 'Y'],Y_training_data_source.loc[~np.isnan(Y_training_data_source['Y']),'Y'],batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

            z_test2 = clf2.predict(S_test.loc[:,vfunc(S.columns)]).reshape(-1)
            
            
            perf_ref=np.append(perf_ref,(sum((z_test-T_test.loc[:,'Y'])**2)+sum((z_test2-S_test.loc[:,'Y'])**2))/(len(z_test)+len(z_test2)))
            
            
            
            z_test = clf.predict(Y_training_data_Target.loc[np.isnan(Y_training_data_Target['Y']),Y_training_data_Target.columns != 'Y']).reshape(-1)
            z_test2 = clf2.predict(Y_training_data_source.loc[np.isnan(Y_training_data_source['Y']),Y_training_data_source.columns != 'Y']).reshape(-1)

            
            perf_ref2=np.append(perf_ref2,(sum((z_test-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Z'])**2)+sum((z_test2-S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Z'])**2))/(len(z_test)+len(z_test2)))
            
    
    #####
    
    
    
    
    
    if Objective_Variable=='both' or Objective_Variable=='discrete' :
    #####COOT######
    ###### DISCRETE ######
        if algo == 'both' or algo =='COOT' :

                          # cost matrix with ot dist
            def compute_cost_matrix(ys,yt,v=10000):
                M=ot.dist(ys.values.reshape(-1,1),yt.values.reshape(-1,1),metric=comp_(v))
                return M  


            if prop_T==0 or type_supervision == 'semi-supervised' :


                if  prop_T==0 :
                    M_lin = None
                else : 
                    M_lin = compute_cost_matrix(yt=Z_training_data_Target['Z'],ys=Z_training_data_source['Z'])


                Ts, Tv, cost = cot_numpy(X1=Z_training_data_source.loc[:,Z_training_data_source.columns != 'Z'], 
                                         X2=Z_training_data_Target.loc[:,Z_training_data_Target.columns != 'Z'], 
                                         niter=100, C_lin=M_lin,
                                                     algo='sinkhorn',reg=1,
                                                     algo2='emd', verbose = False)

                # Target estimation
                enc = onehot(handle_unknown='ignore',sparse=False,categories=[np.arange(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))])
                zs_onehot = enc.fit_transform(S['Z'].values.reshape(-1,1))
                zt_onehot_estimated = len(T.loc[:,'Z'])*np.dot(Ts.T,zs_onehot)
                zt_estimated=enc.inverse_transform(zt_onehot_estimated).reshape(-1)

                
                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target))!=0 :
                    perf_coot=np.append(perf_coot,sum(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z']==zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target)])/len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target)))
                    #perf_tot=sum(T.loc[:,'Z']==zt_estimated)/len(zt_estimated)    
                else : perf_coot = np.append(perf_coot,sum(T.loc[:,'Z']==zt_estimated)/len(zt_estimated))

                    
            #############train classifier and evaluate performance on test
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='sigmoid'),
                    Dense(units=nClass,activation = 'sigmoid')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_size = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_size,)
                loss = 'categorical_crossentropy'
                #loss = 'MeanSquaredError'
                #clf = clf_seq(shape, nClass = len(np.unique(S['Z'])))
                clf = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                
                #print(enc.fit_transform(zt_estimated.reshape(-1,1)))
                #print(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                
                
                clf.fit(T.loc[:,vfunc(T.columns)],enc.fit_transform(zt_estimated.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
                z_test = clf.predict(T_test.loc[:,vfunc(T.columns)])
                z_test = enc.inverse_transform(z_test).reshape(-1)
                perf_coot_test=np.append(perf_coot_test,sum(z_test==T_test.loc[:,'Z'])/len(z_test))
                
                
                
                
            ######################## 


            elif type_supervision=='partial' or type_supervision=='cross-partial'  :


                #Source labelled data learning
                Source_indexes_labelled=np.where(Z_training_data_source['Z']!=-1)[0]

                M_lin = compute_cost_matrix(yt=Z_training_data_Target['Z'],ys=Z_training_data_source.loc[Source_indexes_labelled,'Z'])
                Ts, Tv, cost = cot_numpy(X1=Z_training_data_source.loc[Source_indexes_labelled,Z_training_data_source.columns != 'Z'], 
                                             X2=Z_training_data_Target.loc[:,Z_training_data_Target.columns != 'Z'], 
                                             niter=100, C_lin=M_lin,
                                                         algo='sinkhorn',reg=1,
                                                         algo2='emd', verbose = False)
                    # Target estimation
                enc = onehot(handle_unknown='ignore',sparse=False,categories=[np.arange(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))])
                zs_onehot = enc.fit_transform(S.loc[Source_indexes_labelled,'Z'].values.reshape(-1,1))
                zt_onehot_estimated = len(T.loc[:,'Z'])*np.dot(Ts.T,zs_onehot)
                zt_estimated=enc.inverse_transform(zt_onehot_estimated).reshape(-1)


                #Target labelled data learning
                Target_indexes_labelled=np.where(Z_training_data_Target['Z']!=-1)[0]
                M_lin = compute_cost_matrix(yt=Z_training_data_source['Z'],ys=Z_training_data_Target.loc[Target_indexes_labelled,'Z'])
                Ts, Tv, cost = cot_numpy(X1=Z_training_data_Target.loc[Target_indexes_labelled,Z_training_data_Target.columns != 'Z'], 
                                             X2=Z_training_data_source.loc[:,Z_training_data_source.columns != 'Z'], 
                                             niter=100, C_lin=M_lin,
                                                         algo='sinkhorn',reg=1,
                                                         algo2='emd', verbose = False)
                    # Source estimation
                enc = onehot(handle_unknown='ignore',sparse=False,categories=[np.arange(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))])
                zt_onehot = enc.fit_transform(T.loc[Target_indexes_labelled,'Z'].values.reshape(-1,1))
                zs_onehot_estimated = len(S.loc[:,'Z'])*np.dot(Ts.T,zt_onehot)
                zs_estimated=enc.inverse_transform(zs_onehot_estimated).reshape(-1)


                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target))!=0 and len(np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source))!=0:
                    perf_coot=np.append(perf_coot,(sum(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z']==zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target)])+sum(S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),'Z']==zs_estimated[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source)]))/(len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target))+len(np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source)))) 
                else : perf_coot = np.append(perf_coot,(sum(T.loc[:,'Z']==zt_estimated)+sum(S.loc[:,'Z']==zs_estimated))/(len(zt_estimated)+len(zs_estimated)))          
                
                
                #############train classifier and evaluate the performance on test
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='sigmoid'),
                    Dense(units=nClass,activation = 'sigmoid')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeT = sum(vfunc(T.columns)) #Nombre de variables de Target
                shapeT = (fe_sizeT,)
                loss = 'categorical_crossentropy'
                clfT = clf_seq(shapeT, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfT.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                
                fe_sizeS = sum(vfunc(S.columns)) #Nombre de variables de Target
                shapeS = (fe_sizeS,)
                loss = 'categorical_crossentropy'
                clfS = clf_seq(shapeS, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfS.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                
                
                
                clfT.fit(T.loc[:,vfunc(T.columns)],enc.fit_transform(zt_estimated.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
                clfS.fit(S.loc[:,vfunc(S.columns)],enc.fit_transform(zs_estimated.reshape(-1,1)),batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

                zt_test = clfT.predict(T_test.loc[:,vfunc(T_test.columns)]) 
                zs_test = clfS.predict(S_test.loc[:,vfunc(S_test.columns)]) 
                
                zt_test=enc.inverse_transform(zt_test).reshape(-1)
                zs_test=enc.inverse_transform(zs_test).reshape(-1)
                
                perf_coot_test=np.append(perf_coot_test,(sum(zt_test==T_test.loc[:,'Z'])+sum(zs_test==S_test.loc[:,'Z']))/(len(zs_test)+len(zt_test)))
            ######################## 


    ######JDCOT######  
        ###### DISCRETE ######

        if algo=='both' or algo=='JDCOOT' :
            
            if prop_T >= 0 and type_supervision != "cross-partial" :


                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='relu'),
                    Dense(units=nClass,activation = 'softmax')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_sizeB,)
                loss = 'categorical_crossentropy'
                #loss = 'MeanSquaredError'
                clfB = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfB.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns)) #Nombre de variables de Source
                shape = (fe_sizeA,)
                loss = 'categorical_crossentropy'
                #loss = 'MeanSquaredError'
                clfA = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfA.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])



                def one_hot(y,nClass):

                    m=min(y)
                    if m == -1:
                        if len(np.unique(y)) != 1 :
                            m=np.sort(np.unique(y))[1]

                    Y = np.zeros((len(y),nClass))
                    for i in range(len(y)):
                        if y[i] != -1:
                            Y[i,(y[i]-m).astype(int)]=1
                    return Y

                def one_hot_inv(z_encoded):
                    return np.vectorize(lambda i : np.argmax(z_encoded[i,:]))(np.arange(z_encoded.shape[0]))

                oh_source=one_hot(Z_training_data_source.loc[:,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                oh_target=one_hot(Z_training_data_Target.loc[:,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                #model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                model1,model2,results=jdcot_multitask_classif(modelA=clfA,modelB=clfB,
                                                              XA=np.array(Z_training_data_source.loc[:,Z_training_data_source.columns != 'Z']),
                                                              YA=oh_source,
                                                              XB=np.array(Z_training_data_Target.loc[:,Z_training_data_Target.columns != 'Z']),
                                                              YB=oh_target,
                                                             yAtruth=S['Z'],
                                                             yBtruth=T['Z'],algo='sinkhorn',reg=1,alpha=alpha)



                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target)) !=0:
                    
                    zpred_enc_target=model2.predict(Z_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),Z_training_data_Target.columns != 'Z'])  
                    zpred_target=one_hot_inv(zpred_enc_target)+min(np.unique(T['Z']))

                    if type_supervision != 'partial':
                    
                        
                        perf_jdcoot=np.append(perf_jdcoot,sum(zpred_target==T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z'])/len(zpred_target))
                        zt_test = one_hot_inv(model2.predict(T_test.loc[:,vfunc(T_test.columns)]))+min(np.unique(T['Z']))
                        perf_jdcoot_test=np.append(perf_jdcoot_test,(sum(zt_test==T_test.loc[:,'Z']))/(len(zt_test)))
                
                    #print(zpred_target)
                    #print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z'])
                    if type_supervision == 'partial':
                        zpred_enc_source=model1.predict(Z_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),Z_training_data_source.columns != 'Z'])
                        zpred_source=one_hot_inv(zpred_enc_source)+min(np.unique(S['Z']))
                        
                        perf_jdcoot=np.append(perf_jdcoot,(sum(zpred_source==S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),'Z'])+sum(zpred_target==T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z']))/(len(zpred_source)+len(zpred_target)))
                        zt_test = one_hot_inv(model2.predict(T_test.loc[:,vfunc(T_test.columns)]))+min(np.unique(T['Z']))
                        zs_test = one_hot_inv(model1.predict(S_test.loc[:,vfunc(S_test.columns)]))+min(np.unique(T['Z']))
                        perf_jdcoot_test=np.append(perf_jdcoot_test,(sum(zt_test==T_test.loc[:,'Z'])+sum(zs_test==S_test.loc[:,'Z']))/(len(zt_test)+len(zs_test)))
                
                else : 
                    print("Set automatically to 1")
                    perf_jdcoot=np.append(perf_jdcoot,1)
                    
############################################################
            elif type_supervision=='cross-partial' :


                
                    
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='relu'),
                    Dense(units=nClass,activation = 'softmax')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_sizeB,)
                loss = 'categorical_crossentropy'
                #loss = 'MeanSquaredError'
                clfB = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfB.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns)) #Nombre de variables de Source
                shape = (fe_sizeA,)
                loss = 'categorical_crossentropy'
                #loss = 'MeanSquaredError'
                clfA = clf_seq(shape, nClass = len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                clfA.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])



                def one_hot(y,nClass):
                    y=np.array(y)
                    m=min(y)
                    if m == -1:
                        if len(np.unique(y)) != 1 :
                            m=np.sort(np.unique(y))[1]

                    Y = np.zeros((len(y),nClass))
                    for i in range(len(y)):
                        if y[i] != -1:
                            Y[i,(y[i]-m).astype(int)]=1
                    return Y

                def one_hot_inv(z_encoded):
                    return np.vectorize(lambda i : np.argmax(z_encoded[i,:]))(np.arange(z_encoded.shape[0]))

#Source Model estimation
                oh_source=one_hot(Z_training_data_source.loc[:,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                #no unlabelled anymore
                oh_target=one_hot(Z_training_data_Target.loc[Z_training_data_Target.loc[:,'Z']!=-1,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                #model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                mod,model1,results=jdcot_multitask_classif(modelB=clfA,modelA=clfB,
                                                              XB=np.array(Z_training_data_source.loc[:,Z_training_data_source.columns != 'Z']),
                                                              YB=oh_source,
                                                              XA=np.array(Z_training_data_Target.loc[Z_training_data_Target.loc[:,'Z']!=-1,Z_training_data_Target.columns != 'Z']),
                                                              YA=oh_target,
                                                             yBtruth=S['Z'],
                                                             yAtruth=T.loc[Z_training_data_Target.loc[:,'Z']!=-1,'Z'],algo='sinkhorn',reg=1,alpha=alpha)


#Target Model estimation
                oh_source=one_hot(Z_training_data_source.loc[Z_training_data_source.loc[:,'Z']!=-1,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                #no unlabelled anymore
                oh_target=one_hot(Z_training_data_Target.loc[:,'Z'],len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))
                #model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                mod,model2,results=jdcot_multitask_classif(modelA=clfA,modelB=clfB,
                                                              XA=np.array(Z_training_data_source.loc[Z_training_data_source.loc[:,'Z']!=-1,Z_training_data_source.columns != 'Z']),
                                                              YA=oh_source,
                                                              XB=np.array(Z_training_data_Target.loc[:,Z_training_data_Target.columns != 'Z']),
                                                              YB=oh_target,
                                                             yAtruth=S.loc[Z_training_data_source.loc[:,'Z']!=-1,'Z'],
                                                             yBtruth=T['Z'],algo='sinkhorn',reg=1,alpha=alpha)




                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target)) !=0:
                    
                    zpred_enc_target=model2.predict(Z_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),Z_training_data_Target.columns != 'Z'])  
                    zpred_target=one_hot_inv(zpred_enc_target)+min(np.unique(T['Z']))

                    zpred_enc_source=model1.predict(Z_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),Z_training_data_source.columns != 'Z'])
                    zpred_source=one_hot_inv(zpred_enc_source)+min(np.unique(S['Z']))
                        
                    perf_jdcoot=np.append(perf_jdcoot,(sum(zpred_source==S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),z_labelled_source),'Z'])+sum(zpred_target==T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z']))/(len(zpred_source)+len(zpred_target)))
                    zt_test = one_hot_inv(model2.predict(T_test.loc[:,vfunc(T_test.columns)]))+min(np.unique(T['Z']))
                    zs_test = one_hot_inv(model1.predict(S_test.loc[:,vfunc(S_test.columns)]))+min(np.unique(T['Z']))
                    perf_jdcoot_test=np.append(perf_jdcoot_test,(sum(zt_test==T_test.loc[:,'Z'])+sum(zs_test==S_test.loc[:,'Z']))/(len(zt_test)+len(zs_test)))
                
                else : 
                    print("Set automatically to 1")
                    perf_jdcoot=np.append(perf_jdcoot,1)
                    
                    ############################################################               
                    
                    
                    
    if Objective_Variable=='both' or Objective_Variable=='continuous' :
        #####COOT######
    ###### CONTINUOUS ######
        if algo == 'both' or algo =='COOT' :

                          # cost matrix with ot dist
            def compute_cost_matrix(ys,yt):
                M=ot.dist(ys.values.reshape(-1,1),yt.values.reshape(-1,1),metric=comp_regression()) #comp_reg ? ou comp_
                return M  


            if prop_T==0 or type_supervision == 'semi-supervised' :


                if  prop_T==0 :
                    M_lin = None
                else : 
                    M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'],ys=Y_training_data_source['Y']) 
                
                #plt.imshow(M_lin)

                Ts, Tv, cost = cot_numpy(X1=Y_training_data_source.loc[:,Y_training_data_source.columns != 'Y'], 
                                         X2=Y_training_data_Target.loc[:,Y_training_data_Target.columns != 'Y'], 
                                         niter=100, C_lin=M_lin,
                                                     algo='sinkhorn',reg=1,
                                                     algo2='emd', verbose = False)

                # Target estimation
                #PLUS DE OH ENC
                #Nombre de variables de Target
                zt_estimated = len(T.loc[:,'Y'])*np.dot(Ts.T,S['Y'])
                #print(S['Y'])
                #print(np.dot(Ts.T,S['Y']))
                
                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target))!=0 :
                    #perfs_tot=np.append(perf_coot,sum((zt_estimated-T.loc[:,'Y'])**2)/len(zt_estimated)
                    perf_coot=np.append(perf_coot,sum((zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)]-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])**2)/len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)))
    
                else : perf_coot = np.append(perf_coot,sum((T.loc[:,'Y']-zt_estimated)**2)/len(zt_estimated))

                #print(zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)])
                #print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])
                #print(zt_estimated[np.sort(y_labelled_target)])
                #print(T.loc[np.sort(y_labelled_target),'Y'])
                
                #############train classifier and evaluate the performance on test
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='linear'),
                    Dense(units=nClass,activation = 'linear')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_size = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_size,)
                #loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clf = clf_seq(shape, nClass = 1)
                clf.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                #print(T.loc[:,vfunc(T.columns)])
                clf.fit(T.loc[:,vfunc(T.columns)],zt_estimated,batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
                z_test = clf.predict(T_test.loc[:,vfunc(T_test.columns)])[:,0]
                #print(z_test)
                #print(T_test.loc[:,'Y'])
                perf_coot_test=np.append(perf_coot_test,sum((z_test-T_test.loc[:,'Y'])**2)/len(z_test))
            ######################## 

            if type_supervision == 'partial' or type_supervision == 'cross-partial' :
                                             
                Source_indexes_labelled=np.where(~np.isnan(Y_training_data_source['Y']))[0]
                M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'],ys=Y_training_data_source.loc[Source_indexes_labelled,'Y'])
                Ts, Tv, cost = cot_numpy(X1=Y_training_data_source.loc[Source_indexes_labelled,Y_training_data_source.columns != 'Y'], 
                                             X2=Y_training_data_Target.loc[:,Y_training_data_Target.columns != 'Y'], 
                                             niter=100, C_lin=M_lin,
                                                         algo='sinkhorn',reg=1,
                                                         algo2='emd', verbose = False)
                    # Target estimation
                #PLUS DE OH ENC
                zt_estimated = len(T.loc[:,'Y'])*np.dot(Ts.T,S.loc[Source_indexes_labelled,'Y'])



                #Target labelled data learning
                Target_indexes_labelled=np.where(~np.isnan(Y_training_data_Target['Y']))[0]
                M_lin = compute_cost_matrix(yt=Y_training_data_source['Y'],ys=Y_training_data_Target.loc[Target_indexes_labelled,'Y'])
                Ts, Tv, cost = cot_numpy(X1=Y_training_data_Target.loc[Target_indexes_labelled,Y_training_data_Target.columns != 'Y'], 
                                             X2=Y_training_data_source.loc[:,Y_training_data_source.columns != 'Y'], 
                                             niter=100, C_lin=M_lin,
                                                         algo='sinkhorn',reg=1,
                                                         algo2='emd', verbose = False)
                    # Source estimation
                #PLUS DE OH ENC
                zs_estimated = len(S.loc[:,'Y'])*np.dot(Ts.T,T.loc[Target_indexes_labelled,'Y'])


                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target))!=0 and len(np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source))!=0:
                    perf_coot=np.append(perf_coot,(sum((T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y']-zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)])**2)+sum((S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Y']-zs_estimated[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source)])**2))/(len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target))+len(np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source)))) 
                else : 
                    perf_coot = np.append(perf_coot,(sum((T.loc[:,'Y']-zt_estimated)**2)+sum((S.loc[:,'Y']-zs_estimated)**2))/(len(zt_estimated)+len(zs_estimated)))          
                    #perf_coot=np.append(perf_coot,1)
                    
            #############train classifier and evaluate the performance on test
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='linear'),
                    Dense(units=nClass,activation = 'linear')        ])
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeT = sum(vfunc(T.columns)) #Nombre de variables de Target
                shapeT = (fe_sizeT,)
                loss = 'MeanSquaredError'
                clfT = clf_seq(shapeT, nClass = 1)
                clfT.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                
                fe_sizeS = sum(vfunc(S.columns)) #Nombre de variables de Target
                shapeS = (fe_sizeS,)
                loss = 'MeanSquaredError'
                clfS = clf_seq(shapeS, nClass = 1)
                clfS.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])
                
                
                
                clfT.fit(T.loc[:,vfunc(T.columns)],zt_estimated,batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated
                clfS.fit(S.loc[:,vfunc(S.columns)],zs_estimated,batch_size=10,epochs=20,verbose=0)  # we train the classifier with target data estimated

                zt_test = clfT.predict(T_test.loc[:,vfunc(T_test.columns)])[:,0]
                zs_test = clfS.predict(S_test.loc[:,vfunc(S_test.columns)])[:,0]
                #print(len(zt_test))
                #print(len(zs_test))
                
                perf_coot_test=np.append(perf_coot_test,(sum((zt_test-T_test.loc[:,'Y'])**2)+sum((zs_test-S_test.loc[:,'Y'])**2))/(len(zs_test)+len(zt_test)))
            ######################## 

                    
            
    #####CONTINUOUS######
        if algo=='both' or algo=='JDCOOT' :
      
                
            if prop_T >= 0 and type_supervision != "cross-partial": #all cases

                """
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='relu'),
                    Dense(units=nClass,activation = 'softmax')        ])
                    return model
                """
                def clf_seq(shape):
                    model = tf_keras.Sequential([Dense(units=128,input_shape=shape,activation='linear'),
                    Dense(units=1,activation = 'linear')]) 
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_sizeB,)
                #loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfB = clf_seq(shape)
                clfB.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns)) #Nombre de variables de Source
                shape = (fe_sizeA,)
                #loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfA = clf_seq(shape)
                clfA.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])


                #Y_training_data_source.loc[Y_training_data_source['Y']==-1,'Y']=np.NaN
                #Y_training_data_Target.loc[Y_training_data_Target['Y']==-1,'Y']=np.NaN

                """
                jdcot multi-task for multi regression problems
                npreds : number of parameters to predict (it has to be the same number for both datasets)
                yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
                YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels and true values if observed labels (semi supervision)
                """
                #model1,model2,results=jdcot_multitask_reg(modelA,modelB,XA,YA,XB,YB,yAtruth,yBtruth)
                model1,model2,results=jdcot_multitask_reg(modelA=clfA,modelB=clfB,
                                                              XA=np.array(Y_training_data_source.loc[:,Y_training_data_source.columns != 'Y']),
                                                              YA=np.array(Y_training_data_source['Y']).reshape((-1,1)),
                                                              XB=np.array(Y_training_data_Target.loc[:,Y_training_data_Target.columns != 'Y']),
                                                              YB=np.array(Y_training_data_Target['Y']).reshape((-1,1)),
                                                              yAtruth=S['Y'],
                                                              yBtruth=T['Y'],reshape_data=False,algo='sinkhorn',reg=100,alpha=alpha)



                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)) !=0:
                    zpred_target=model2.predict(Y_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),Y_training_data_Target.columns != 'Y'])  
                    zpred_target=zpred_target.ravel()
                    
                    if type_supervision != 'partial' : 
                        perf_jdcoot=np.append(perf_jdcoot,sum((zpred_target-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])**2)/len(zpred_target))

                    #print(zpred_target)
                    #print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])
                    if type_supervision == 'partial':
                        zpred_source=model1.predict(Y_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),Y_training_data_source.columns != 'Y'])
                        zpred_source=zpred_source.ravel()
                        perf_jdcoot=np.append(perf_jdcoot,(sum((zpred_source-S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Y'])**2)+sum((zpred_target-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])**2))/(len(zpred_source)+len(zpred_target)))
                else : 
                    print("Set automatically to 1")
                    perf_jdcoot=np.append(perf_jdcoot,1)

                ###ON TEST
                zt_test = model2.predict(T_test.loc[:,vfunc(T_test.columns)])[:,0]
                zs_test = model1.predict(S_test.loc[:,vfunc(S_test.columns)])[:,0]
                #print(zs_test)
                #print(sum((zs_test-S_test.loc[:,'Z'])**2)/len(zs_test))
                if type_supervision == 'partial' : 
                    perf_jdcoot_test=np.append(perf_jdcoot_test,(sum((zt_test-T_test.loc[:,'Y'])**2)+sum((zs_test-S_test.loc[:,'Y'])**2))/(len(zt_test)+len(zs_test)))

                else : perf_jdcoot_test=np.append(perf_jdcoot_test,(sum((zt_test-T_test.loc[:,'Y'])**2))/(len(zt_test)))

        ############################################################
            elif type_supervision=='cross-partial' :
                
                
                def clf_seq(shape):
                    model = tf_keras.Sequential([Dense(units=128,input_shape=shape,activation='linear'),
                    Dense(units=1,activation = 'linear')]) 
                    return model

                vfunc = np.vectorize(lambda arr : 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns)) #Nombre de variables de Target
                shape = (fe_sizeB,)
                #loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfB = clf_seq(shape)
                clfB.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns)) #Nombre de variables de Source
                shape = (fe_sizeA,)
                #loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfA = clf_seq(shape)
                clfA.compile(optimizer='Adam',loss=loss,metrics=['accuracy'])

#Source model
                
                mod,model1,results=jdcot_multitask_reg(modelB=clfA,modelA=clfB,
                                                              XB=np.array(Y_training_data_source.loc[:,Y_training_data_source.columns != 'Y']),
                                                              YB=np.array(Y_training_data_source['Y']).reshape((-1,1)),
                                                              XA=np.array(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target.loc[:,'Y']),Y_training_data_Target.columns != 'Y']),
                                                              YA=np.array(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target.loc[:,'Y']),'Y']).reshape((-1,1)),
                                                              yBtruth=S['Y'],
                                                              yAtruth=T.loc[~np.isnan(Y_training_data_Target.loc[:,'Y']),'Y'],reshape_data=False,algo='sinkhorn',reg=100,alpha=alpha)
#Target Model 
                mod,model2,results=jdcot_multitask_reg(modelA=clfA,modelB=clfB,
                                                              XA=np.array(Y_training_data_source.loc[~np.isnan(Y_training_data_source.loc[:,'Y']),Y_training_data_source.columns != 'Y']),
                                                              YA=np.array(Y_training_data_source.loc[~np.isnan(Y_training_data_source.loc[:,'Y']),'Y']).reshape((-1,1)),
                                                              XB=np.array(Y_training_data_Target.loc[:,Y_training_data_Target.columns != 'Y']),
                                                              YB=np.array(Y_training_data_Target['Y']).reshape((-1,1)),
                                                              yAtruth=S.loc[~np.isnan(Y_training_data_source.loc[:,'Y']),'Y'],
                                                              yBtruth=T['Y'],reshape_data=False,algo='sinkhorn',reg=100,alpha=alpha)


                if len(np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)) !=0:
                    zpred_target=model2.predict(Y_training_data_Target.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),Y_training_data_Target.columns != 'Y'])  
                    zpred_target=zpred_target.ravel()

                    zpred_source=model1.predict(Y_training_data_source.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),Y_training_data_source.columns != 'Y'])
                    zpred_source=zpred_source.ravel()
                    perf_jdcoot=np.append(perf_jdcoot,(sum((zpred_source-S.loc[np.setdiff1d(np.arange(0,np.shape(S)[0]),y_labelled_source),'Y'])**2)+sum((zpred_target-T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])**2))/(len(zpred_source)+len(zpred_target)))
                else : 
                    print("Set automatically to 1")
                    perf_jdcoot=np.append(perf_jdcoot,1)

                ###ON TEST
                zt_test = model2.predict(T_test.loc[:,vfunc(T_test.columns)])[:,0]
                zs_test = model1.predict(S_test.loc[:,vfunc(S_test.columns)])[:,0]
                #print(zs_test)
                #print(sum((zs_test-S_test.loc[:,'Z'])**2)/len(zs_test))
                perf_jdcoot_test=np.append(perf_jdcoot_test,(sum((zt_test-T_test.loc[:,'Y'])**2)+sum((zs_test-S_test.loc[:,'Y'])**2))/(len(zt_test)+len(zs_test)))

               
        ############################################################
        
        
        
        
            
            

            
        
    #print(perf_coot_test)
    #print(perf_jdcoot_test)
    if len(perf_coot) == 0 : perf_coot = np.NaN
    if len(perf_jdcoot) == 0 : perf_jdcoot = np.NaN
    if len(perf_coot_test) == 0 : perf_coot_test = np.NaN
    if len(perf_jdcoot_test) == 0 : perf_jdcoot_test = np.NaN
        
        
    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
    print("\n")
    print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
    print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
    print("\n")
    print("Pure Performance Reference : {} ".format(perf_ref2))
    print("Test Performance Reference : {} ".format(perf_ref))
 
        
    return(perf_coot,perf_jdcoot,perf_coot_test,perf_jdcoot_test,perf_ref,perf_ref2)

# -

# # Experiments and Graphics 

# ### Data observation variation

# +
"""
Semi_Supervised_Labelled_Proportion_Variations : Compute and plot performance versus the proportion of labelled observations in target

Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            OR
                            
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            If None, Learning data is splited into 30% of test data and 70% of learning data
                            
                            
        Proportion_Labelled_Variations : np array of values of labelled proportions to vary
        
        Monte_Carlo : Number of repetitions of computing
                            
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        



Output : Data (tuple) : tuple of 10  with respectively : Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT

"""


def Semi_Supervised_Labelled_Proportion_Variations(Data,Data_test=None,Proportion_Labelled_Variations=np.array([0,0.5,0.9]),Monte_Carlo=1,algo='both',Objective_Variable='discrete',Balance=False,alpha=None):
    
    nPoints=len(Proportion_Labelled_Variations)
    proportions=Proportion_Labelled_Variations
    Store_in = True
    
    if Objective_Variable=='both' : 
        return -1
    
    if Monte_Carlo >= 1 :
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
        else :
        
            MC_coot_perf=np.repeat(0,nPoints)
            MC_jdcoot_perf=np.repeat(0,nPoints)
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,nPoints)

            test_MC_coot_perf=np.repeat(0,nPoints)
            test_MC_jdcoot_perf=np.repeat(0,nPoints)
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,nPoints)


            

            with alive_bar(Monte_Carlo*nPoints, bar="classic2", spinner="twirls") as bar:
                for i in range(Monte_Carlo):
                    print("Entering Monte Carlo loop number {}".format(i))
                    coot_perf=np.array([])
                    jdcoot_perf=np.array([])
                    ref_perf=np.array([])
                    test_coot_perf=np.array([])
                    test_jdcoot_perf=np.array([])
                    test_ref_perf=np.array([])

                    for prop in proportions :
                        perf=Performance(Data,Data_test=Data_test,alpha=alpha,algo=algo ,type_supervision='semi-supervised',Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=prop )
                        coot_perf= np.append(coot_perf,perf[0])
                        jdcoot_perf=np.append(jdcoot_perf,perf[1])
                        ref_perf=np.append(ref_perf,perf[5])

                        test_coot_perf= np.append(test_coot_perf,perf[2])
                        test_jdcoot_perf=np.append(test_jdcoot_perf,perf[3])
                        test_ref_perf=np.append(test_ref_perf,perf[4])




                        bar()
                    MC_coot_perf=MC_coot_perf+coot_perf
                    MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                    MC_ref_perf=MC_ref_perf+ref_perf
                    bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                    bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)

                    test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                    test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                    test_MC_ref_perf=test_MC_ref_perf+test_ref_perf
                    test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                    test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                    
                MC_ref_perf=MC_ref_perf/Monte_Carlo
                MC_coot_perf=MC_coot_perf/Monte_Carlo
                MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

                test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
                test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
                test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
                
        
                   
        if Objective_Variable == "discrete":
            tit2="Accuracy"
        else : tit2 ="MSE"

        if algo=='both' or algo=='COOT' : 
            
            tit="Test Accuracy of methods versus the proportion of target data labelled" 
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].plot(np.arange(nPoints)+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(nPoints)], patch_artist = True,
           boxprops = dict(facecolor = "red"))
            
            tit="Pure Accuracy of methods versus the proportion of target data observed" 
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[1].set_xlabel("Proportion of target observation labelled")
            axs[0].plot(np.arange(nPoints)+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(nPoints)],patch_artist = True,
           boxprops = dict(facecolor = "red"))
            
        if algo=='both' or algo=='JDCOOT' : 
            
            tit="Test Accuracy of methods versus the proportion of target data labelled" 
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].plot(np.arange(nPoints)+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(nPoints)],patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            tit="Pure Accuracy of methods versus the proportion of target data labelled" 
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[1].set_xlabel("Proportion of target observation labelled")
            axs[0].plot(np.arange(nPoints)+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(nPoints)],patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            
        axs[0].plot(np.arange(nPoints)+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(nPoints)+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        
        axs[1].legend(loc="best",fontsize="5")
        axs[0].legend(loc="best",fontsize="5")
            
        axs[0].set_xticks(np.arange(nPoints)+1, labels=np.around(proportions,2))
        axs[1].set_xticks(np.arange(nPoints)+1, labels=np.around(proportions,2))
        
        plt.tight_layout()
        plt.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Semi_Supervised_Labelled_Proportion_Variations")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
    





    



# +
"""
Partial_Labelled_Proportion_Variations : Compute and plot performance versus the proportion of labelled observations in target and source

Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            OR
                            
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            If None, Learning data is splited into 30% of test data and 70% of learning data
                            
                            
        Proportion_Labelled_Variations : np array of values of labelled proportions to vary
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""
def Partial_Labelled_Proportion_Variations(Data,Data_test=None,Proportion_Labelled_Variations=np.array([0.1,0.5,0.9]),Monte_Carlo=1,algo='both',Objective_Variable='discrete',Multi_Variations=True ,Balance=False,alpha=None ):
    Sample_sizes=Proportion_Labelled_Variations #pratique
    Store_in = True
    if Objective_Variable=='both' : 
        return -1
    
    

    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :
        
            

            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref_perf=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref_perf=np.array([])
                for samp_s in Proportion_Labelled_Variations :
                    for samp_t in Proportion_Labelled_Variations :


                        perfs=Performance(Data, Data_test,alpha=alpha,algo=algo ,type_supervision='partial',Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=samp_t,Labelled_Proportion_Source=samp_s )
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])

                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref_perf=np.append(ref_perf,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref_perf=np.append(test_ref_perf,perfs[4])




                MC_ref_perf=MC_ref_perf+ref_perf
                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)

                test_MC_ref_perf=test_MC_ref_perf+test_ref_perf
                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)

                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Observed labels proportion variation impact","$p^T=p^S=$Proportion of \n observed labels","Observed \n labels proportion")
        
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot),"Partial_Labelled_Proportion_Variations")

        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                hm_coot,
                hm_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    
                    perfs=Performance(Data, Data_test,alpha=alpha,algo=algo ,type_supervision='partial',Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=samp_s,Labelled_Proportion_Source=samp_s )
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])


                MC_ref_perf=MC_ref_perf+ref
                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


        if algo=='JDCOOT' : 
            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        if algo == 'both' :
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        
        axs[1].legend(loc="best",fontsize="5")
        
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Partial_Labelled_Proportion_Variations")

        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
    


# +
"""
Cross_Partial_Labelled_Proportion_Variations : Compute and plot performance versus the proportion of labelled observations in target and source, but uses cross partial JDCOOT method

Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            OR
                            
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            If None, Learning data is splited into 30% of test data and 70% of learning data
                            
                            
        Proportion_Labelled_Variations : np array of values of labelled proportions to vary
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""
def Cross_Partial_Labelled_Proportion_Variations(Data,Data_test=None,Proportion_Labelled_Variations=np.array([0.1,0.5,0.9]),Monte_Carlo=1,algo='both',Objective_Variable='discrete',Multi_Variations=True ,Balance=False,alpha=None ):
    Sample_sizes=Proportion_Labelled_Variations #pratique
    Store_in = True
    if Objective_Variable=='both' : 
        return -1
    
    

    
    if Multi_Variations :
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :
        
            

            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref_perf=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref_perf=np.array([])
                for samp_s in Proportion_Labelled_Variations :
                    for samp_t in Proportion_Labelled_Variations :


                        perfs=Performance(Data, Data_test,alpha=alpha,algo=algo ,type_supervision='cross-partial',Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=samp_t,Labelled_Proportion_Source=samp_s )
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])
                        
                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref_perf=np.append(ref_perf,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref_perf=np.append(test_ref_perf,perfs[4])




                MC_ref_perf=MC_ref_perf+ref_perf
                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)

                test_MC_ref_perf=test_MC_ref_perf+test_ref_perf
                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Observed labels proportion variation impact","$p^T=p^S=$Proportion of \n observed labels","Observed \n labels proportion")
        


        
        if Store_in :
            Store(Data,"Cross_Partial_Labelled_Proportion_Variations")

        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    
                    perfs=Performance(Data, Data_test,alpha=alpha,algo=algo ,type_supervision='cross-partial',Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=samp_s,Labelled_Proportion_Source=samp_s )
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])


                MC_ref_perf=MC_ref_perf+ref
                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


        if algo=='JDCOOT' : 
            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        if algo == 'both' :
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Test Accuracy vs Observed labels \n proportion variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs Observed labels \n proportion variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        
        axs[1].legend(loc="best",fontsize="5")
        
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Cross_Partial_Labelled_Proportion_Variations")

        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
  

# +
"""
Observed_Labels_Proportions_Variation : Compute and plot performance versus the proportion of labelled observations in target and/or source. Permit to choose what type of supervision

Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            OR
                            
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                        | X_1 | ... | X_d | Y | Z |
                                        
                                    with X_i the ith observed covariate, 
                                         Y the continuous objective variable for regression analysis (optional), 
                                         Z the discrete objective variable for classification analysis (optional)
                                         
                            If None, Learning data is splited into 30% of test data and 70% of learning data
                            
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        Proportion_Labelled_Variations : np array of values of labelled proportions to vary
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""
def Observed_Labels_Proportions_Variation(Data,Data_test=None,type_supervision='unsupervised',Proportion_Labelled_Variations=np.array([0.1,0.5,0.9]),Monte_Carlo=1,algo='both',Objective_Variable='discrete',Multi_Variations=True ,Balance=False,alpha=None):
    if type_supervision == 'partial':
        return Partial_Labelled_Proportion_Variations(Data,Data_test,Proportion_Labelled_Variations,Monte_Carlo,algo,Objective_Variable,Multi_Variations ,Balance,alpha)
    
    elif type_supervision == 'unsupervised' : 
        return Semi_Supervised_Labelled_Proportion_Variations(Data,Data_test,np.array([0]),Monte_Carlo,algo,Objective_Variable,Balance,alpha)

    elif type_supervision == 'semi-supervised' :
        return Semi_Supervised_Labelled_Proportion_Variations(Data,Data_test,Proportion_Labelled_Variations,Monte_Carlo,algo,Objective_Variable,Balance,alpha)
    
    elif type_supervision == 'cross-partial':
        return Cross_Partial_Labelled_Proportion_Variations(Data,Data_test,Proportion_Labelled_Variations,Monte_Carlo,algo,Objective_Variable,Multi_Variations ,Balance,alpha)
    
                  
# -

# ### Data Generation variation

# +
"""
Sample_Size_Variation : Compute and plot performance versus the sample sizes

Input : Sample_Size_Variation (np array) : samples sizes values to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""

def Sample_Size_Variation(Sample_sizes,Data=None,Monte_Carlo=1,Multi_Variations=True , algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None  
                            ,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, Sparse_Rate=0.75,
                          Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                          Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):
    
    if Objective_Variable=='both' : 
        return -1
    
    
    
    Store_in = True
    
    test_Data_=data_generator(300, 300, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate, 
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target, 
                               1,1,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)

    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :



            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :
                        scenario= data_generator(samp_s, samp_t, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate, 
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target, 
                               Observed_Covariates_Proportion_Source,Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)



                        perfs=Performance(scenario, test_Data_,alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )

                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])


                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Sample size variation impact","$n^T=n^S=$Sample Size","sample size")
        


        
        if Store_in : 
            Store(Data,"Sample_Size_Variation")

        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    scenario= data_generator(samp_s, samp_s, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target, 
                               Observed_Covariates_Proportion_Source,Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)




                    perfs=Performance(scenario,test_Data_, alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )

                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs \n sample size variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$n^T=n^S=$Sample Size")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs \n sample size variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$n^T=n^S=$Sample Size")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


        if algo=='JDCOOT' : 
            tit="Test Accuracy vs \n sample size variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$n^T=n^S=$Sample Size")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n sample size variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$n^T=n^S=$Sample Size")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        if algo == 'both' :
            tit="Test Accuracy vs \n sample size variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$n^T=n^S=$Sample Size")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$n^T=n^S=$Sample Size")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$n^T=n^S=$Sample Size")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n sample size variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$n^T=n^S=$Sample Size")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Sample_Size_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
        
        
                  

# +
"""
Proportion_Of_Observed_Covariates_Variation : Compute and plot performance versus the proportion of observed covariates

Input : Observed_Covariates_Proportion (np array) : proportion of observed covariates values to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""



def Proportion_Of_Observed_Covariates_Variation(Observed_Covariates_Proportion,Data=None, Monte_Carlo=1,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None   
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, Sparse_Rate=0.75,
                            Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,Indexes_Chosen_For_Generation=INDEX_GENERATION):
    
    
    
    
    
    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=Observed_Covariates_Proportion
    Store_in=True
    
    
    test_Data_ = data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target,
                               Observed_Covariates_Proportion_Source=1, Observed_Covariates_Proportion_Target=1,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)

    
    

    
    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :




            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :
                        scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target,
                               Observed_Covariates_Proportion_Source=samp_s, Observed_Covariates_Proportion_Target=samp_t,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)


                        perfs=Performance(scenario, test_Data_,alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )

                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])


                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))

                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Proportion of observed covariables variation impact","$n^T=n^S=$Proportion of \n observed covariables","Proportion of \n observed covariables")
        



        
        if Store_in : 
            Store(Data,"Proportion_Of_Observed_Covariates_Variation")

        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source, R2_Target,
                               Observed_Covariates_Proportion_Source=samp_s, Observed_Covariates_Proportion_Target=samp_s,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Poisson=Poisson)




                    perfs=Performance(scenario,test_Data_, alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )

                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs Proportion of \n observed covariables variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs Proportion of \n observed covariables variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


        if algo=='JDCOOT' : 
            tit="Test Accuracy vs Proportion of \n observed covariables variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs Proportion of \n observed covariables variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        if algo == 'both' :
            tit="Test Accuracy vs Proportion of \n observed covariables variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs Proportion of observed \n covariables variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^x_T=p^x_S=$Proportion of \n observed covariables")
            axs[0].plot(np.arange(len(Sample_sizes))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(Sample_sizes))],patch_artist = True,
           boxprops = dict(facecolor = "green"))



        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Proportion_Of_Observed_Covariates_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
        
        
                  
        
                  

# +
"""
R2_Variation : Compute and plot performance versus the R2 coefficient

Input : R2 (np array) : R2 values to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""


def R2_Variation(R2 ,Data=None,Monte_Carlo=1,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None   
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, Sparse_Rate=0.75,
                         Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):

    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=R2
    Store_in =True
    ######################### test samples generation : 
    ##
    ex=Sref()
    Col_observed_target=ex[1].columns[:-2]
    Col_observed_source=ex[0].columns[:-2]
    ex=None
    ##
    tests1=[]
    tests2=[]
    for samp_s in Sample_sizes :
        for samp_t in Sample_sizes :       
            scenario_test = data_generator(math.ceil(n_Source*0.3), math.ceil(n_Target*0.3), d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                           mean_Y_Target, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                           Odds_Ratio_Source, Odds_Ratio_Target, R2_Source=samp_s, R2_Target=samp_t,
                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, 
                                           Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                           Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=Col_observed_source,
                                           Cols_Chosen_For_Observation_Target=Col_observed_target,Poisson=Poisson
                                          )
                    

            tests1.append(scenario_test)
            if samp_s==samp_t : 
                    tests2.append(scenario_test)
                        
        #####################""
    

    
    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :



            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :

                        scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source=samp_s, R2_Target=samp_t,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                                 Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests1[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests1[j][1].columns[:-2],Poisson=Poisson)


                        perfs=Performance(scenario, tests1[j],alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                        j=j+1
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])


                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))

                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"$R^2$ variation impact","$R^2_T=R^2_S$","$R^2$")
        



       
        if Store_in :
            Store(Data,"R2_Variation")

        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source, Odds_Ratio_Target, R2_Source=samp_s, R2_Target=samp_s,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                             Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],Poisson=Poisson)




                    perfs=Performance(scenario,tests2[j], alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                    j=j+1
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs \n $R^2$ variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$R^2_T=R^2_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs \n $R^2$ variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$R^2_T=R^2_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            

        if algo=='JDCOOT' : 
            tit="Test Accuracy vs \n $R^2$ variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$R^2_T=R^2_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n $R^2$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$R^2_T=R^2_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
           
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))




        if algo == 'both' :
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$R^2_T=R^2_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Accuracy vs \n $R^2$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$R^2_T=R^2_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            tit="Test Accuracy vs \n $R^2$ variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$R^2_T=R^2_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n $R^2$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$R^2_T=R^2_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        


        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"R2_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
        
        
                  

# +
"""
OR_Variation : Compute and plot performance versus the Odds Ratio coefficient

Input : OR (np array) : OR values to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""


def OR_Variation(OR,Data=None,Monte_Carlo=1,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None   
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, Sparse_Rate=0.75,
                            R2_Source=0.6, R2_Target=0.6,Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):

    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=OR
    R2=Sample_sizes
    Store_in = True
    ######################### test samples generation : 
    ##
    ex=Sref()
    Col_observed_target=ex[1].columns[:-2]
    Col_observed_source=ex[0].columns[:-2]
    ex=None
    ##
    tests1=[]
    tests2=[]
    for samp_s in Sample_sizes :
        for samp_t in Sample_sizes :       
            scenario_test = data_generator(math.ceil(n_Source*0.3), math.ceil(n_Target*0.3), d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                           mean_Y_Target, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                           Odds_Ratio_Source=samp_s, Odds_Ratio_Target=samp_t, R2_Source=R2_Source, R2_Target=R2_Source,
                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                           Cols_Chosen_For_Observation_Source=Col_observed_source,Cols_Chosen_For_Observation_Target=Col_observed_target,Poisson=Poisson)

            tests1.append(scenario_test)
            if samp_s==samp_t : 
                tests2.append(scenario_test)
                        
        #####################""
    

    
    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :


            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :
                        scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source=samp_s, Odds_Ratio_Target=samp_t, R2_Source=R2_Source, R2_Target=R2_Source,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests1[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests1[j][1].columns[:-2],Poisson=Poisson)




                        perfs=Performance(scenario, tests1[j],alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                        j=j+1
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])

                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])

                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))

                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"$OR$ variation impact","$OR_T=OR_S$","$OR$")
        




        
        if Store_in :
            Store(Data,"OR_Variation")
        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source=samp_s, Odds_Ratio_Target=samp_s, R2_Source=R2_Source, R2_Target=R2_Target,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],Poisson=Poisson)


                    perfs=Performance(scenario,tests2[j], alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                    j=j+1
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if algo=='COOT' : 
            tit="Test Accuracy vs \n $OR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$OR_T=OR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs \n $OR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$R^2_T=R^2_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
           
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            

        if algo=='JDCOOT' : 
            tit="Test Accuracy vs \n $OR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$OR_T=OR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n $OR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$OR_T=OR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))




        if algo == 'both' :
            tit="Test Accuracy vs \n $OR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$OR_T=OR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$OR_T=OR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))



            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$OR_T=OR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs $OR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$OR_T=OR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
           
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))


        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"OR_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
        
                  

# +
"""
Sparse_Rate_Variation : Compute and plot performance versus the Sparse rate

Input : SR (np array) : SR values to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                                    
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""



def Sparse_Rate_Variation(SR,Data=None,Monte_Carlo=1 ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None   
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, R2_Source=0.6, R2_Target=0.6,Odds_Ratio_Source=0.5,Odds_Ratio_Target=0.5,Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):

    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=SR
    R2=Sample_sizes
    Store_in = True
    ######################### test samples generation : 
    ##
    ex=Sref()
    Col_observed_target=ex[1].columns[:-2]
    Col_observed_source=ex[0].columns[:-2]
    ex=None
    ##
    j=0
    tests1=[]
    tests2=[]
    for samp_s in Sample_sizes :
        for samp_t in Sample_sizes :       
            scenario_test = data_generator(math.ceil(n_Source*0.3), math.ceil(n_Target*0.3), d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                           mean_Y_Target, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, Sparse_Rate=samp_s, 
                           Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Source,
                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, 
                                           Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                           Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                           Cols_Chosen_For_Observation_Source=Col_observed_source,
                                           Cols_Chosen_For_Observation_Target=Col_observed_target,Poisson=Poisson)
                                          
                                          
                    
            tests1.append(scenario_test)
            if samp_s==samp_t : 
                tests2.append(scenario_test)
                        
        #####################""
    

    if True : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :


            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate=samp_s,
                               Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Target,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],Poisson=Poisson)



                    perfs=Performance(scenario,tests2[j], alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                    j=j+1
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        if Objective_Variable == "discrete":
            tit2="Accuracy"
        else : tit2 ="MSE"
            
            
        if algo=='COOT' : 
            tit="Test Accuracy vs \n $SR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].set_xlabel("$SR_T=SR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs \n $SR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[0].set_xlabel("$SR_T=SR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
           
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            

        if algo=='JDCOOT' : 
            tit="Test Accuracy vs \n $SR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].set_xlabel("$SR_T=SR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n $SR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[0].set_xlabel("$SR_T=SR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance",patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))])




        if algo == 'both' :
            tit="Test Accuracy vs \n $SR$ variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].set_xlabel("$SR_T=SR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))


            axs[0].set_ylabel(tit2)
            axs[0].set_xlabel("$SR_T=SR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))



            axs[1].set_ylabel(tit2)
            axs[1].set_xlabel("$SR_T=SR_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n $SR$ variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[0].set_xlabel("$SR_T=SR_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Sparse_Rate_Variation")

        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)



# +
"""
Mean_Shift_Variation : Compute and plot performance versus the mean shift between target and source

Input : Mean_Shift (np array) : Mean Shift values between target and source to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""



def Mean_Shift_Variation(Mean_Shift,Data=None ,Monte_Carlo=1,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None 
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0, rho_source_generation=0.7,rho_source_non_generation=0.2,
                          rho_target_generation=0.7,rho_target_non_generation=0.2, Sparse_Rate=0.75,
                           Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):

    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=Mean_Shift
    R2=Sample_sizes
    Store_in = True
    
    
    
    ######################### test samples generation : 
    ##
    ex=Sref()
    Col_observed_target=ex[1].columns[:-2]
    Col_observed_source=ex[0].columns[:-2]
    ex=None
    ##
    tests1=[]
    tests2=[]
    for samp_s in Sample_sizes :
        for samp_t in Sample_sizes :       
            scenario_test = data_generator(math.ceil(n_Source*0.3), math.ceil(n_Target*0.3), d_Source, d_Target, mean_X_Source+samp_s, mean_X_Target+samp_t, mean_Y_Source,
                           mean_Y_Target, rho_source_generation,rho_source_non_generation,
                          rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                           Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Source,
                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=Col_observed_source,Cols_Chosen_For_Observation_Target=Col_observed_target,Poisson=Poisson)


            tests1.append(scenario_test)
            if samp_s==R2[0]: 
                tests2.append(scenario_test)
                        
        #####################""
    

    
    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
            
        else :


            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :
                        scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source+samp_s, mean_X_Target+samp_t, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Source,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests1[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests1[j][1].columns[:-2],Poisson=Poisson)




                        perfs=Performance(scenario, tests1[j],alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                        j=j+1
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])


                        if samp_s==R2[0] :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))

                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Mean shift variation impact","$i \ s.t. \mu_T=\mu_S+i$ \n with $\mu_S$ = " + str(R2[0]),"mean shift")



        
        

        
        if Store_in :
            Store(Data,"Mean_Shift_Variation")

        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target+samp_s, mean_Y_Source,
                               mean_Y_Target, rho_source_generation,rho_source_non_generation,
                              rho_target_generation,rho_target_non_generation, Sparse_Rate,  
                               Odds_Ratio_Source=samp_s, Odds_Ratio_Target=samp_s, R2_Source=R2_Source, R2_Target=R2_Target,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],Poisson=Poisson)



                    perfs=Performance(scenario,tests2[j], alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                    j=j+1
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        

        if algo=='COOT' : 
            tit="Test Accuracy vs \n mean shift variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs \n mean shift variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            

        if algo=='JDCOOT' : 
            tit="Test Accuracy vs v mean shift variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n mean shift variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with \mu_S = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))




        if algo == 'both' :
            tit="Test Accuracy vs v mean shift variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
           
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))



            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs \n mean shift variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            
        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Mean_Shift_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)

# +
"""
Correlation_Variation : Compute and plot performance versus auto correlation coefficient of covariates in source and target

Input : Correlation (np array) : auto correlation values between target and source to compute performance with

        Data (tuple or str) : 
                            (str) Name of the folder where data has been stored
                             
                            OR 
                            
                            Previous simulated data (tuple of 10) with respectively : 
                                                        Pure performance Monte Carlo COOT,
                                                         Pure performance Monte Carlo JDCOOT,
                                                         Pure performance Monte Carlo REF,
                                                         Pure performance Boxplot COOT,
                                                         Pure performance Boxplot JDCOOT,
                                                         Test performance Monte Carlo COOT,
                                                         Test performance Monte Carlo JDCOOT,
                                                         Test performance Monte Carlo REF,
                                                         Test performance Boxplot COOT,
                                                         Test performance Boxplot JDCOOT
                                         
        
                            
        type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                       if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                       if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                       if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset
        
        
        
        Monte_Carlo : Number of repetitions of computing
                            
                            
        Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps
        
        
        Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation 
                                                                if 'continuous', measure the performance for regression
        
        algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT 
                                                    if 'JDCOOT', measure the performance for JDCOOT
                                                    if 'both',  measure the performance for COOT and JDCOOT
                                                    
        Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data. 
        
        alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())
        
        Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes
        
        Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.
        
        Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis. 
        
        d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")
        
        mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target
        
        mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target
        
        Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source
        
        Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target
        
        Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)
        
        Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)
        
        R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)
        
        Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target
        
        Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly 
        




Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Pure performance Heatmap COOT,
                                                                        Pure performance Heatmap JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT,
                                                                        Test performance Heatmap COOT,
                                                                        Test performance Heatmap JDCOOT,
                                                                


                                                                    For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                        Pure performance Monte Carlo COOT,
                                                                        Pure performance Monte Carlo JDCOOT,
                                                                        Pure performance Monte Carlo REF,
                                                                        Pure performance Boxplot COOT,
                                                                        Pure performance Boxplot JDCOOT,
                                                                        Test performance Monte Carlo COOT,
                                                                        Test performance Monte Carlo JDCOOT,
                                                                        Test performance Monte Carlo REF,
                                                                        Test performance Boxplot COOT,
                                                                        Test performance Boxplot JDCOOT
    
"""



def Correlation_Variation(Correlation,Data=None ,Monte_Carlo=1,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Objective_Variable='discrete',Balance=True,alpha=None,Poisson=False, Labelled_Proportion_Target=None,
                          Labelled_Proportion_Source=None  
                            ,n_Source=1000,n_Target=1000,d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), 
                          mean_X_Target=np.zeros(100), 
                          mean_Y_Source=0,mean_Y_Target=0,rho_source_non_generation=0.2
                          ,rho_target_non_generation=0.2,Sparse_Rate=0.75,
                           R2_Source=0.6, R2_Target=0.6,Odds_Ratio_Source=0.5,Odds_Ratio_Target=0.5,Observed_Covariates_Proportion_Target=0.2, 
                          Observed_Covariates_Proportion_Source=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION):

    if Objective_Variable=='both' : 
        return -1
    
    Sample_sizes=Correlation
    R2=Sample_sizes
    Store_in = True
     ######################### test samples generation : 
    ##
    ex=Sref()
    Col_observed_target=ex[1].columns[:-2]
    Col_observed_source=ex[0].columns[:-2]
    ex=None
    ##
    tests1=[]
    tests2=[]
    for samp_s in Sample_sizes :
        for samp_t in Sample_sizes :       
            scenario_test = data_generator(math.ceil(n_Source*0.3), math.ceil(n_Target*0.3), d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                           mean_Y_Target, Source_Generation_Correlation=samp_s,Source_Non_Generation_Correlation=rho_source_non_generation,
                          Target_Generation_Correlation=samp_t,Target_Non_Generation_Correlation=rho_target_non_generation, Sparse_Rate=Sparse_Rate,
                           Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Source,
                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, 
                                           Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                           Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                          Cols_Chosen_For_Observation_Source=Col_observed_source,
                                          Cols_Chosen_For_Observation_Target=Col_observed_target,Poisson=Poisson)



            tests1.append(scenario_test)
            if samp_s==samp_t : 
                tests2.append(scenario_test)
                        
        #####################""
    

    
    if Multi_Variations :
        
        if isinstance(Data,str) : 
            Data = Read(Data)
            Store_in = False
        
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==14:
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else : 
                print("Wrong data format")
                return 0 
        else :


            test_hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            test_hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            hm_coot=np.zeros((len(Sample_sizes),len(Sample_sizes)))
            hm_jdcoot=np.zeros((len(Sample_sizes),len(Sample_sizes)))

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])

            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])

            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))
            MC_ref_perf=np.repeat(0,len(Sample_sizes))


            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot=np.array([])
                test_z_jdcoot=np.array([])
                z_coot=np.array([])
                z_jdcoot=np.array([])
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])

                ref=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    for samp_t in Sample_sizes :
                        scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, Source_Generation_Correlation=samp_s,Source_Non_Generation_Correlation=rho_source_non_generation,
                              Target_Generation_Correlation=samp_t,Target_Non_Generation_Correlation=rho_target_non_generation, Sparse_Rate=Sparse_Rate, 
                               Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Source,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests1[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests1[j][1].columns[:-2],Poisson=Poisson)



                        perfs=Performance(scenario, tests1[j],alpha=alpha,algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                        j=j+1
                        z_coot=np.append(z_coot,perfs[0])
                        z_jdcoot=np.append(z_jdcoot,perfs[1])
                        test_z_coot=np.append(test_z_coot,perfs[2])
                        test_z_jdcoot=np.append(test_z_jdcoot,perfs[3])


                        if samp_t==samp_s :
                            coot_perf= np.append(coot_perf,perfs[0])
                            jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                            ref=np.append(ref,perfs[5])

                            test_coot_perf= np.append(test_coot_perf,perfs[2])
                            test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                            test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)



                hm_coot=hm_coot+z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                hm_jdcoot=hm_jdcoot+z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))



                test_hm_coot=test_hm_coot+test_z_coot.reshape((len(Sample_sizes),len(Sample_sizes)))
                test_hm_jdcoot=test_hm_jdcoot+test_z_jdcoot.reshape((len(Sample_sizes),len(Sample_sizes)))
                
                
                
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo
            

            hm_coot=hm_coot/Monte_Carlo
            hm_jdcoot=hm_jdcoot/Monte_Carlo
            test_hm_coot=test_hm_coot/Monte_Carlo
            test_hm_jdcoot=test_hm_jdcoot/Monte_Carlo
            
            
            Data=(MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot)
                
                
        graph_multi(Data,algo,Sample_sizes,Objective_Variable,"Correlation variation impact","$rho_T=rho_S$","correlation")


        
        


        
        if Store_in :
            Store(Data,"Correlation_Variation")
        return(Data)
        
    else : 
        
        
        
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        if isinstance(Data,str) : 
            
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data)!=2) :
            if len(Data)==10 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot=Data
                Store_in = False
            elif len(Data) == 14 :
                MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,hm_coot,hm_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot,test_hm_coot,test_hm_jdcoot=Data
                Store_in = False
            else :
                print("Wrong data format")
                return 0 
            
        else :

            MC_coot_perf=np.repeat(0,len(Sample_sizes))
            MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            bp_coot=np.array([[]])
            bp_jdcoot=np.array([[]])
            MC_ref_perf=np.repeat(0,len(Sample_sizes))
            test_MC_coot_perf=np.repeat(0,len(Sample_sizes))
            test_MC_jdcoot_perf=np.repeat(0,len(Sample_sizes))
            test_bp_coot=np.array([[]])
            test_bp_jdcoot=np.array([[]])
            test_MC_ref_perf=np.repeat(0,len(Sample_sizes))



            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf=np.array([])
                jdcoot_perf=np.array([])
                ref=np.array([])

                test_coot_perf=np.array([])
                test_jdcoot_perf=np.array([])
                test_ref=np.array([])
                j=0
                for samp_s in Sample_sizes :
                    scenario= data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source, mean_X_Target, mean_Y_Source,
                               mean_Y_Target, Source_Generation_Correlation=samp_s,Source_Non_Generation_Correlation=rho_source_non_generation,
                              Target_Generation_Correlation=samp_s,Target_Non_Generation_Correlation=rho_target_non_generation, Sparse_Rate=Sparse_Rate,
                               Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source, R2_Target=R2_Target,
                               Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],Poisson=Poisson)



                    perfs=Performance(scenario, tests2[j],alpha=alpha, algo=algo ,type_supervision=type_supervision,Objective_Variable=Objective_Variable,Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,Labelled_Proportion_Source=Labelled_Proportion_Source )
                    j=j+1
                    coot_perf= np.append(coot_perf,perfs[0])
                    jdcoot_perf=np.append(jdcoot_perf,perfs[1])
                    ref=np.append(ref,perfs[5])

                    test_coot_perf= np.append(test_coot_perf,perfs[2])
                    test_jdcoot_perf=np.append(test_jdcoot_perf,perfs[3])
                    test_ref=np.append(test_ref,perfs[4])



                MC_coot_perf=MC_coot_perf+coot_perf
                MC_jdcoot_perf=MC_jdcoot_perf+jdcoot_perf
                MC_ref_perf=MC_ref_perf+ref
                bp_coot=np.concatenate((bp_coot,np.array([coot_perf])),axis=1)
                bp_jdcoot=np.concatenate((bp_jdcoot,np.array([jdcoot_perf])),axis=1)


                test_MC_coot_perf=test_MC_coot_perf+test_coot_perf
                test_MC_jdcoot_perf=test_MC_jdcoot_perf+test_jdcoot_perf
                test_MC_ref_perf=test_MC_ref_perf+test_ref
                test_bp_coot=np.concatenate((test_bp_coot,np.array([test_coot_perf])),axis=1)
                test_bp_jdcoot=np.concatenate((test_bp_jdcoot,np.array([test_jdcoot_perf])),axis=1)
                
            MC_ref_perf=MC_ref_perf/Monte_Carlo
            MC_coot_perf=MC_coot_perf/Monte_Carlo
            MC_jdcoot_perf=MC_jdcoot_perf/Monte_Carlo
            

            test_MC_ref_perf=test_MC_ref_perf/Monte_Carlo
            test_MC_coot_perf=test_MC_coot_perf/Monte_Carlo
            test_MC_jdcoot_perf=test_MC_jdcoot_perf/Monte_Carlo



        

        

        if algo=='COOT' : 
            tit="Test Accuracy vs covariables shift \n correlation variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$rho_T=rho_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Pure Accuracy vs covariables shift \n correlation variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$rho_T=rho_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            

        if algo=='JDCOOT' : 
            tit="Test Accuracy vs covariables shift \n correlation variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$rho_T=rho_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs covariables shift \n correlation variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$rho_T=rho_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
           
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))




        if algo == 'both' :
            tit="Test Accuracy vs covariables shift \n correlation variations" 
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$rho_T=rho_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_coot_perf,color="red",label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))

            tit="Accuracy vs covariables shift \n correlation variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$rho_T=rho_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_coot_perf,color="red",label="COOT Mean Performance")
            
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "red"))



            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$rho_T=rho_S$")
            axs[1].plot(np.arange(len(R2))+1,test_MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))

            tit="Pure Accuracy vs covariables shift \n correlation variations" 
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$rho_T=rho_S$")
            axs[0].plot(np.arange(len(R2))+1,MC_jdcoot_perf,color="green",label="JDCOOT Mean Performance")
            
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo,-1))[:,i] for i in range(len(R2))],patch_artist = True,
           boxprops = dict(facecolor = "green"))
            
            
        axs[0].plot(np.arange(len(Sample_sizes))+1,MC_ref_perf,'--',color="blue",label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes))+1,test_MC_ref_perf,'--',color="blue",label="Reference")
        

        axs[0].legend(loc="best",fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes))+1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        #fig.show()
        if Store_in :
            Store((MC_coot_perf,MC_jdcoot_perf,MC_ref_perf,bp_coot,bp_jdcoot,test_MC_coot_perf,test_MC_jdcoot_perf,test_MC_ref_perf,test_bp_coot,test_bp_jdcoot),"Correlation_Variation")
        return(MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
                  
