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

# # $\alpha$ Optimization

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

# %matplotlib inline

# OT

# NN
# import keras # for DL, from install tensorflow
# from keras import layers

# ML

# MISC

# PERSO
# import cot # COOT
# import jdcot # JDCOOT
# import classif
pd.options.mode.chained_assignment = None  # default='warn'
# -

global INDEX_GENERATION
INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
# %run JDCOOT_FUNCTIONS.ipynb #Functions import
from jdcoot import *


# # $\alpha$ optimization for Classification

def optim_alpha(range_alpha, Objective_Variable='discrete', acc=1):
    ALPHA = np.zeros(3)
    P = np.zeros(3)
    n_S = 1000
    n_T = 1000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    sr_S = sr
    sr_T = sr
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    REF = data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T + 0.3, mY_S, mY_T, rho_source_generation - 0.2,
                         rho_source_non_generation,
                         rho_target_generation, rho_target_non_generation, sr_S - 0.05, sr_T + 0.05, OR_S, OR_T + 0.1,
                         R2_S, R2_T + 0.1,
                         pxo_S, pxo_T + 0.1, INDEX_GENERATION)
    Col_observed_target = REF[1].columns[:-2]
    Col_observed_source = REF[0].columns[:-2]
    TEST = data_generator(300, 300, d_S, d_T, mX_S, mX_T + 0.3, mY_S, mY_T, rho_source_generation - 0.2,
                          rho_source_non_generation,
                          rho_target_generation, rho_target_non_generation, sr_S - 0.05, sr_T + 0.05, OR_S, OR_T + 0.1,
                          R2_S, R2_T + 0.1,
                          1, 1, INDEX_GENERATION, Cols_Chosen_For_Observation_Source=Col_observed_source,
                          Cols_Chosen_For_Observation_Target=Col_observed_target)

    for i in range(acc):
        perfs = np.zeros(3)
        a = np.zeros(3)

        for alp in range_alpha:

            """
            p=Performance(REF,TEST,type_supervision='unsupervised',Objective_Variable=Objective_Variable,Balance=True,alpha=alp)
            p1=np.mean(p[1])
            p2=np.mean(p[3])
            if (p1+p2)/2 > perfs[0]:
                a[0]=alp
                perfs[0]=(p1+p2)/2
            """
            p = Observed_Labels_Proportions_Variation(REF, TEST, 'semi-supervised', np.array([0.1, 0.5, 0.9]),
                                                      Monte_Carlo=1, algo='JDCOOT', Balance=True, alpha=alp)
            p1 = np.mean(p[4])
            p2 = np.mean(p[9])
            if (p1 + p2) / 2 > perfs[1]:
                a[1] = alp
                perfs[1] = (p1 + p2) / 2
            """  
            p=Observed_Labels_Proportions_Variation(REF, TEST,'partial',np.array([0.1,0.5,0.9]),Monte_Carlo=1,algo='JDCOOT',Multi_Variations=False,Balance=True,alpha=alp)
            p1=np.mean(p[4])
            p2=np.mean(p[9])
            if (p1+p2)/2 > perfs[2]:
                a[2]=alp
                perfs[2]=(p1+p2)/2
            """

        ALPHA = ALPHA + a
        P = perfs + P

    return (ALPHA / acc, P / acc)


# #### Alpha for unsupervised

optimized_alpha = optim_alpha(np.array([0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7]),
                              Objective_Variable='discrete', acc=10)
print(optimized_alpha)
# 0.661

# #### Alpha for semi-supervised

optimized_alpha = optim_alpha(np.array([3.25, 3.3, 3.35, 3.4, 3.45]), Objective_Variable='discrete', acc=10)
print(optimized_alpha)
# 3.335

# #### Alpha for partial

optimized_alpha = optim_alpha(np.array([2.75, 2.8, 2.85, 2.9, 2.95]), Objective_Variable='discrete', acc=10)
print(optimized_alpha)


# 2.875

# # $\alpha$ optimization for Regression

def optim_alpha(range_alpha, Objective_Variable='continuous', acc=1):
    ALPHA = np.zeros(3)
    P = np.zeros(3)
    n_S = 1000
    n_T = 1000
    d = 100
    d_S = d
    d_T = d
    pxo = 0.2
    pxo_S = pxo
    pxo_T = pxo
    sr = 0.75
    sr_S = sr
    sr_T = sr
    OR_S = 0.5
    OR_T = 0.5
    R2_S = 0.6
    R2_T = 0.6
    mX_S = np.zeros(d_S)
    mX_T = np.zeros(d_T)
    mY_S = 0
    mY_T = 0
    # cov_S = np.eye(d_S)
    # cov_T = np.eye(d_T)
    rho_source_generation = 0.7
    rho_source_non_generation = 0.2
    rho_target_generation = 0.7
    rho_target_non_generation = 0.2

    REF = data_generator(n_S, n_T, d_S, d_T, mX_S, mX_T + 0.3, mY_S, mY_T, rho_source_generation - 0.2,
                         rho_source_non_generation,
                         rho_target_generation, rho_target_non_generation, sr_S - 0.05, sr_T + 0.05, OR_S, OR_T + 0.1,
                         R2_S - 0.1, R2_T + 0.1,
                         pxo_S, pxo_T + 0.1, INDEX_GENERATION)
    Col_observed_target = REF[1].columns[:-2]
    Col_observed_source = REF[0].columns[:-2]
    print(len(Col_observed_target))
    print(len(Col_observed_source))
    TEST = data_generator(300, 300, d_S, d_T, mX_S, mX_T + 0.3, mY_S, mY_T, rho_source_generation - 0.2,
                          rho_source_non_generation,
                          rho_target_generation, rho_target_non_generation, sr_S - 0.05, sr_T + 0.05, OR_S, OR_T + 0.1,
                          R2_S - 0.1, R2_T + 0.1,
                          1, 1, INDEX_GENERATION, Cols_Chosen_For_Observation_Source=Col_observed_source,
                          Cols_Chosen_For_Observation_Target=Col_observed_target)

    for i in range(acc):
        perfs = np.zeros(3)
        perfs = perfs + 10000
        a = np.zeros(3)

        for alp in range_alpha:

            """
            p=Performance(REF,TEST,type_supervision='unsupervised',Objective_Variable=Objective_Variable,Balance=False,alpha=alp)
            p1=np.mean(p[1])
            p2=np.mean(p[3])
            if (p1+p2)/2 < perfs[0]:
                a[0]=alp
                perfs[0]=(p1+p2)/2
           
            print(alp)
            p=Observed_Labels_Proportions_Variation(REF, TEST,'semi-supervised',np.array([0.1,0.5,0.9]),Objective_Variable=Objective_Variable,Monte_Carlo=1,algo='JDCOOT',Balance=False,alpha=alp)
            p1=np.mean(p[4])
            p2=np.mean(p[9])
            print(p1)
            print(p2)
            if (p1+p2)/2 < perfs[1]:
                a[1]=alp
                perfs[1]=(p1+p2)/2
            """
            p = Observed_Labels_Proportions_Variation(REF, TEST, 'partial', np.array([0.1, 0.5, 0.9]),
                                                      Objective_Variable=Objective_Variable, Monte_Carlo=1,
                                                      algo='JDCOOT', Multi_Variations=False, Balance=False, alpha=alp)
            p1 = np.mean(p[4])
            print(p1)
            p2 = np.mean(p[9])
            print(p2)
            if (p1 + p2) / 2 < perfs[2]:
                a[2] = alp
                perfs[2] = (p1 + p2) / 2

        ALPHA = ALPHA + a
        P = perfs + P

    return (ALPHA / acc, P / acc)


# #### Alpha for unsupervised

optimized_alpha = optim_alpha(np.array([0, 0.1, 0.2, 0.4, 0.6, 0.8]), Objective_Variable='continuous', acc=2)
print(optimized_alpha)
# 0.3

# #### Alpha for semi-supervised

optimized_alpha = optim_alpha(np.array([2.5, 2.75, 3]), Objective_Variable='continuous', acc=2)
print(optimized_alpha)
# 2.625

# #### Alpha for partial

optimized_alpha = optim_alpha(np.array([2.4, 2.45, 2.5, 2.55, 2.6]), Objective_Variable='continuous', acc=2)
print(optimized_alpha)
# 2.425
