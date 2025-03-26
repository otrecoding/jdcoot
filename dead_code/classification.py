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

import math

# BASICS
import numpy as np  # scientific computing
import pandas as pd
import scipy.io  # Input and output
# ML
from sklearn import preprocessing

# %matplotlib inline
# OT
# NN
# import keras # for DL, from install tensorflow
# from keras import layers

# MISC

# PERSO
# import cot # COOT
# import jdcot # JDCOOT
# import classif
pd.options.mode.chained_assignment = None  # default='warn'
# -

# %run Codes_Lucas\JDCOOT_FUNCTIONS.ipynb #Functions import
from jdcoot import *

# # Data Generation

# ### Same variables for generation, to stay in the "same world"

global INDEX_GENERATION
INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

# # Reference Scenario function

test_Data = Sref_test(INDEX_GENERATION)
reference_Data = Sref(INDEX_GENERATION)

Poisson_Data_Reference = Sref_Poisson(INDEX_GENERATION)
Poisson_Data_Test = Sref_Poisson_test(INDEX_GENERATION)

# # Test on Categorial artificial Data 

# ## Test of Performance function

# ### Independent covariables

# +
D = data_generator(n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100),
                   mean_X_Target=np.zeros(100), mean_Y_Source=0,
                   mean_Y_Target=0, Source_Generation_Correlation=0, Source_Non_Generation_Correlation=0,
                   Target_Generation_Correlation=0, Target_Non_Generation_Correlation=0, Sparse_Rate=0.75,
                   Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                   Observed_Covariates_Proportion_Source=0.2, Observed_Covariates_Proportion_Target=0.2,
                   Indexes_Chosen_For_Generation=INDEX_GENERATION)

Performance(D, test_Data, type_supervision='semi-supervised', Objective_Variable='discrete', Balance=True, alpha=None,
            Labelled_Proportion_Target=0.2)
# -

# ### Auto Correlated covariables 

Performance(reference_Data, test_Data, type_supervision='unsupervised', Objective_Variable='discrete', Balance=False,
            alpha=None, Labelled_Proportion_Target=0.2)

# ## Data observation impact

# ### Learning cases impact

# #### Unsupervised

UN = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'unsupervised', Monte_Carlo=10, algo='both',
                                           Balance=False, alpha=None)

UN = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'unsupervised', Monte_Carlo=10, algo='both',
                                           Balance=True, alpha=None)

# #### Semi-supervised

SS = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                           np.array([0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]), Monte_Carlo=10,
                                           algo='both', Balance=False, alpha=None)

SS = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                           np.array([0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]), Monte_Carlo=10,
                                           algo='both', Balance=True, alpha=None)

SS = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                           np.array([0.1, 0.3, 0.5, 0.7, 0.9]), Monte_Carlo=10, algo='both',
                                           Balance=False, alpha=None)

SS = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'semi-supervised',
                                           np.array([0.1, 0.3, 0.5, 0.7, 0.9]), Monte_Carlo=10, algo='both',
                                           Balance=True, alpha=None)

# With more categories

P = Observed_Labels_Proportions_Variation(Poisson_Data_Reference, Poisson_Data_Test, 'semi-supervised',
                                          np.array([0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]), Monte_Carlo=5, algo='both',
                                          Balance=True, alpha=None)

# #### Partial and Cross-Partial

PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'partial',
                                           Proportion_Labelled_Variations=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                           Monte_Carlo=5, algo='both', Objective_Variable='discrete',
                                           Multi_Variations=True, Balance=False, alpha=None)

PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'partial',
                                           Proportion_Labelled_Variations=np.array([0.02, 0.05, 0.07, 0.1]),
                                           Monte_Carlo=5, algo='both', Objective_Variable='discrete',
                                           Multi_Variations=True, Balance=False, alpha=None)

PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'cross-partial',
                                           Proportion_Labelled_Variations=np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                           Monte_Carlo=5, algo='both', Objective_Variable='discrete',
                                           Multi_Variations=True, Balance=False, alpha=None)

PA = Observed_Labels_Proportions_Variation(reference_Data, test_Data, 'cross-partial',
                                           Proportion_Labelled_Variations=np.array([0.02, 0.05, 0.07, 0.1]),
                                           Monte_Carlo=5, algo='both', Objective_Variable='discrete',
                                           Multi_Variations=True, Balance=False, alpha=None)

# ### Proportion of observed covariables variations

PXO = Proportion_Of_Observed_Covariates_Variation(Observed_Covariates_Proportion=np.array([0.2, 0.4, 0.6, 0.8]),
                                                  Monte_Carlo=10, Multi_Variations=True, algo='both',
                                                  type_supervision='unsupervised', Balance=False, alpha=None)

# ## Data Generation variations

# ### Sample size variations

SSV = Sample_Size_Variation(Sample_sizes=np.array([10, 100, 500, 1000]), Monte_Carlo=10, Multi_Variations=True,
                            algo='both', type_supervision='unsupervised', Balance=False, alpha=None)

# ### OR variations

OR = OR_Variation(OR=np.array([0.2, 0.4, 0.6, 0.8]), Monte_Carlo=10, Multi_Variations=True, algo='both',
                  type_supervision='unsupervised', Balance=False, alpha=None)

# ### Sparse rate variations

SR = Sparse_Rate_Variation(SR=np.array([0.25, 0.5, 0.75, 1]), Monte_Carlo=10, algo='both',
                           type_supervision='unsupervised', Balance=False, alpha=None)

# ### Mean shift

# Unbalanced
MS = Mean_Shift_Variation(Mean_Shift=np.array([0, 0.1, 0.2, 0.3, 0.4]), Monte_Carlo=10, Multi_Variations=True,
                          algo='both', type_supervision='unsupervised', Balance=False, alpha=None)

# Balanced
MS = Mean_Shift_Variation(Mean_Shift=np.array([0, 0.1, 0.2, 0.3, 0.4]), Monte_Carlo=10, Multi_Variations=True,
                          algo='both', type_supervision='unsupervised', Balance=True, alpha=None)

# ### Correlation variations

C = Correlation_Variation(Correlation=np.array([0, 0.2, 0.5, 0.7, 1]), Monte_Carlo=10, Multi_Variations=True,
                          algo='both', type_supervision='unsupervised', Balance=False, alpha=None)

# # Test on CaffeNet4096 and GoogleNet1024 Data

# ## Data importation

# +
tests = []
data_source = {}
data_target = {}

min_max_scaler = preprocessing.MinMaxScaler()
# Collab
possible_data = scipy.io.loadmat('caltech10_caffe.mat')
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
S_data = [feat, labels]
S_nClass = len(np.unique(labels))  # nb de class in source data
possible_data = scipy.io.loadmat('amazon_google.mat')
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
T_data = [feat, labels]
T_nClass = len(np.unique(labels))  # nb de class in target data

S = pd.DataFrame(np.concatenate((S_data[0], S_data[1].reshape(-1, 1)), axis=1),
                 columns=['X' + str(i) for i in range(S_data[0].shape[1])] + ['Z'])
T = pd.DataFrame(np.concatenate((T_data[0], T_data[1].reshape(-1, 1)), axis=1),
                 columns=['X' + str(i) for i in range(T_data[0].shape[1])] + ['Z'])
# -

# ## Performance function test

S = pd.DataFrame(np.concatenate((S_data[0], S_data[1].reshape(-1, 1)), axis=1),
                 columns=['X' + str(i) for i in range(S_data[0].shape[1])] + ['Z'])
T = pd.DataFrame(np.concatenate((T_data[0], T_data[1].reshape(-1, 1)), axis=1),
                 columns=['X' + str(i) for i in range(T_data[0].shape[1])] + ['Z'])
Performance((S, T), type_supervision='semi-supervised', algo="both", Balance=False, Labelled_Proportion_Target=0.2)

# ## Data labelling impact

UN = Observed_Labels_Proportions_Variation((S, T), None, 'unsupervised', Monte_Carlo=10, algo='both', Balance=False)

SS = Observed_Labels_Proportions_Variation((S, T), None, 'semi-supervised',
                                           np.array([0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]), Monte_Carlo=10,
                                           algo='both', Balance=False)

SS = Observed_Labels_Proportions_Variation((S, T), None, 'semi-supervised', np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                           Monte_Carlo=10, algo='both', Balance=False)

PA = Observed_Labels_Proportions_Variation((S, T), None, 'partial', np.array([0.1, 0.3, 0.5, 0.7, 0.9]), Monte_Carlo=4,
                                           algo='both', Balance=False)

PA = Observed_Labels_Proportions_Variation((S, T), None, 'partial', np.array([0.02, 0.05, 0.07, 0.1]), Monte_Carlo=4,
                                           algo='both', Balance=False)

PA = Observed_Labels_Proportions_Variation((S, T), None, 'cross-partial', np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                                           Monte_Carlo=4, algo='both', Balance=False)

PA = Observed_Labels_Proportions_Variation((S, T), None, 'cross-partial', np.array([0.02, 0.05, 0.07, 0.1]),
                                           Monte_Carlo=4, algo='both', Balance=False)
