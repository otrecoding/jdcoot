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

from jdcoot import *

# PERSO
#import cot # COOT
#import jdcot # JDCOOT
#import classif
pd.options.mode.chained_assignment = None  # default='warn'
# -

# %run Codes_Lucas\JDCOOT_FUNCTIONS.ipynb #Functions import

# # Data Generation

# ### Same variables for generation, to stay in the "same world"

global INDEX_GENERATION
INDEX_GENERATION=np.random.choice(np.arange(100), math.ceil(0.75 * 100),replace=False)

# # Reference Scenario function

test_Data=Sref_test(INDEX_GENERATION)
reference_Data=Sref(INDEX_GENERATION)

# # Test on Continuous artificial Data 

# ## Test of Performance function

# ### Independent covariables

D=data_generator(n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100), mean_X_Target=np.zeros(100), mean_Y_Source=0,
                           mean_Y_Target=0, Source_Generation_Correlation=0,Source_Non_Generation_Correlation=0,
                          Target_Generation_Correlation=0,Target_Non_Generation_Correlation=0, Sparse_Rate=0.75,
                           Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                           Observed_Covariates_Proportion_Source=0.2, Observed_Covariates_Proportion_Target=0.2,Indexes_Chosen_For_Generation=INDEX_GENERATION)
Performance(D,test_Data,type_supervision='semi-supervised',Objective_Variable='continuous',Balance=False,alpha=None,Labelled_Proportion_Target=0.2)

# ### Auto Correlated covariables 

Performance(reference_Data,test_Data,type_supervision='unsupervised',Objective_Variable='continuous',Balance=False,alpha=None,Labelled_Proportion_Target=0.2)

# ## Data Observation variations

# ### Learning cases impact

# #### Unsupervised

UN=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'unsupervised',Monte_Carlo=10,Objective_Variable='continuous',algo='both',Balance=False,alpha=None)

# #### Semi-supervised

SE=Observed_Labels_Proportions_Variation(reference_Data, test_Data,'semi-supervised',np.array([0,0.02,0.05,0.07,0.1,0.12,0.15]),Monte_Carlo=10,Objective_Variable='continuous',algo='both',Balance=False,alpha=None)

SE=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'semi-supervised',np.array([0.1, 0.3, 0.5, 0.7, 0.9]),Monte_Carlo=10,Objective_Variable='continuous',algo='both',Balance=False,alpha=None)

# #### Partial and Cross-Partial

PA=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'partial',Proportion_Labelled_Variations=np.array([0.1,0.3,0.5,0.7,0.9]),Monte_Carlo=5,algo='both',Objective_Variable='continuous',Multi_Variations=True ,Balance=False,alpha=None)

PA=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'partial',Proportion_Labelled_Variations=np.array([0.02,0.05,0.07,0.1]),Monte_Carlo=5,algo='both',Objective_Variable='continuous',Multi_Variations=True ,Balance=False,alpha=None)

PA=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'cross-partial',Proportion_Labelled_Variations=np.array([0.1,0.3,0.5,0.7,0.9]),Monte_Carlo=5,algo='both',Objective_Variable='continuous',Multi_Variations=True ,Balance=False,alpha=None )

PA=Observed_Labels_Proportions_Variation(reference_Data,test_Data,'cross-partial',Proportion_Labelled_Variations=np.array([0.02,0.05,0.07,0.1]),Monte_Carlo=5,algo='both',Objective_Variable='continuous',Multi_Variations=True ,Balance=False,alpha=None )

# ### Proportion of observed covariables variations

PXO=Proportion_Of_Observed_Covariates_Variation(Observed_Covariates_Proportion=np.array([0.2,0.4,0.6,0.8]),Monte_Carlo=10,Objective_Variable='continuous',Multi_Variations=True ,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)

# ## Data Generation variations

# ### Sample size variations

SSV=Sample_Size_Variation(Sample_sizes=np.array([10,100,500,1000]),Monte_Carlo=10 ,Objective_Variable='continuous',Multi_Variations=True,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)

# ### $R^2$ variations

R2=R2_Variation(R2=np.array([0.2,0.4,0.6,0.8]),Monte_Carlo=10,Objective_Variable='continuous',Multi_Variations=True ,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)

# ### Sparse rate variations

SR=Sparse_Rate_Variation(SR=np.array([0.25,0.5,0.75,1]),Objective_Variable='continuous',Monte_Carlo=10,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)

# ### Mean shift

MS=Mean_Shift_Variation(Mean_Shift=np.array([0,0.1,0.2,0.3,0.4]),Objective_Variable='continuous',Monte_Carlo=10,Multi_Variations=True,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)

# ### Correlation variations

C=Correlation_Variation(Correlation=np.array([0,0.2,0.5,0.7,1]),Objective_Variable='continuous',Monte_Carlo=10,Multi_Variations=True ,algo='both',type_supervision='unsupervised',Balance=False,alpha=None)
