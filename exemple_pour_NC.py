# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: jdcoot
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import sklearn
import numpy as np
import pandas as pd
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import math
from scipy.io import loadmat

from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_partial_jdcoot2 import discrete_partial_jdcoot2 # fonction pour verifier que jdcoot donne les memes perfs que coot quand la boucle n'est que sur le transport
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_partial_coot2 import discrete_partial_coot2 # 2 coot semi supervisé
from jdcoot.models.discrete_partial_jdcoot3 import discrete_partial_jdcoot3 # 2 coot plus JDCOOT
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference
from sklearn.model_selection import train_test_split
# -

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# # les donnees caffeNet GoogleNet

# +
featuresToUse = ["CaffeNet4096", "GoogleNet1024"] 
#featuresToUse = ["CaffeNet4096", "CaffeNet4096"] 
sourceDomainName = ['caltech10'] #['caltech10','amazon','webcam']
targetDomainName = ['caltech10'] #['caltech10','amazon','webcam']

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# Collab
possible_data = loadmat(os.path.join("data/", "features", featuresToUse[0],
                                                 "caltech10" + '.mat'))
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
S_data = [feat, labels]
S_nClass = len(np.unique(labels)) # nb de class in source data
possible_data = loadmat(os.path.join("data/", "features", featuresToUse[1],
                                                 "amazon" + '.mat'))
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
T_data = [feat, labels]
T_nClass = len(np.unique(labels)) # nb de class in target data

source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1

# -

results = []
numRepetitions = 10
alpha =1.5# hyperparamètre devant la loss a été optimé
prop_target = 0.005
algo = "sinkhorn"
reg = 1


# +
a = np.random.choice(np.arange(len(source)), math.ceil(0.8 * len(source)), replace=False)
b = np.random.choice(np.arange(len(target)), math.ceil(0.8 * len(target)), replace=False)

S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
S = source.iloc[a, :].reset_index(drop=True)
T = target.iloc[b, :].reset_index(drop=True)
# -

S.shape, T.shape, S_test.shape, T_test.shape

# # Non supervisé

pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(
                S, T, S_test, T_test,algo="sinkhorn",reg=1,batch_size=20,alpha =1.5
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_unsupervised_coot(
                S, T, S_test, T_test,algo="sinkhorn",reg=1,batch_size=20
            )
test_target

# # semi supervisé

n_target = len(T.Z)
l_train, l_test = train_test_split(
        np.arange(n_target),
        train_size=0.01,
    )

pure_source, pure_target, test_source, test_target = discrete_semisupervised_jdcoot(
                S, T, S_test, T_test,l_train, l_test,algo="sinkhorn",reg=1,batch_size=20,alpha =0.001         
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_semisupervised_coot(
                S, T, S_test, T_test,l_train, l_test,algo="sinkhorn",reg=1,batch_size=20            
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_semisupervised_reference(
                S, T, S_test, T_test,l_train, l_test,algo="sinkhorn",reg=1,batch_size=20                      
            )
test_target

# # partial

# +
# label observé aléatoire dans source et target

# +
prop_source=0.5
prop_target=0.005 
n_target = len(T.Z)
n_source = len(S.Z)
l_source_train, l_source_test = train_test_split(
        np.arange(n_source),
        train_size=prop_source
    )

l_target_train, l_target_test = train_test_split(
        np.arange(n_target),
        train_size=prop_target
    )
# -

l_target_train

# +
# lun label dans chaque classe dans source et target

# +

# SOURCE
l_source_train = []
for z in np.unique(S['Z']):
    idx = np.where(S['Z'] == z)[0]
    chosen = np.random.choice(idx, size=1, replace=False)
    l_source_train.extend(chosen)

l_source_train = np.array(l_source_train)

# le reste en test
l_source_test = np.setdiff1d(np.arange(len(S['Z'])), l_source_train)


# TARGET
l_target_train = []
for z in np.unique(T['Z']):
    idx = np.where(T['Z'] == z)[0]
    chosen = np.random.choice(idx, size=1, replace=False)
    l_target_train.extend(chosen)

l_target_train = np.array(l_target_train)

l_target_test = np.setdiff1d(np.arange(len(T['Z'])), l_target_train)

# +
# les labels 0,1,2,3,4 observés dans source et les labels 5,6,7,8,9 observés dans source

# +
import numpy as np

# ===== SOURCE =====
l_source_train = np.where(np.isin(S['Z'], [0,1,2,3,4]))[0]
l_source_test  = np.where(np.isin(S['Z'], [5,6,7,8,9]))[0]

# ===== TARGET =====
l_target_train = np.where(np.isin(T['Z'], [5,6,7,8,9]))[0]
l_target_test  = np.where(np.isin(T['Z'], [0,1,2,3,4]))[0]
# -

pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20            
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_coot2(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20          

            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot2(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20,alpha =2
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20,alpha =1.5
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot3(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20,alpha =1.5
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test
            )
test_target


# +
import numpy as np
# Exemple de valeurs à tester pour alpha
alpha_values = np.linspace(1, 3, 6)  # 0, 0.1, 0.2, ..., 1.0
best_alpha = None
best_score = -np.inf  # ou 0 selon ta métrique
results = []
numRepetitions = 1
for repe in range(numRepetitions):
   
        a = np.random.choice(np.arange(len(source)), math.ceil(0.8 * len(source)), replace=False)
        b = np.random.choice(np.arange(len(target)), math.ceil(0.8 * len(target)), replace=False)

        S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
        T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
        S = source.iloc[a, :].reset_index(drop=True)
        T = target.iloc[b, :].reset_index(drop=True)
        n_target = len(T.Z)
        n_source = len(S.Z)
        l_source_train, l_source_test = train_test_split(
            np.arange(n_source),
            train_size=prop_source
        )

        l_target_train, l_target_test = train_test_split(
            np.arange(n_target),
            train_size=prop_target
            )
        for a in alpha_values:
            pure_source, pure_target, test_source, test_target = \
            discrete_partial_jdcoot(S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="sinkhorn",reg=1,batch_size=20,alpha =a)

            score = test_target  
    
            if score > best_score:
                best_score = score
                best_alpha = a

        results.append({
         "repetition": repe,
         "recoding": "jdcoot",
         "learning": "unsupervised",
         "alpha": best_alpha,
     })

   

df_results = pd.DataFrame(results)


df_results

df_summary = (
    df_results
    .groupby(
        ["recoding", "learning", "alpha"],
        as_index=False
    )
    .agg(
        alpha_mean=("alpha", "mean"),
    )
)

df_summary

df_summary.to_excel("results_mean.xlsx", index=False)

# -

df_summary.to_excel("results_mean.xlsx", index=False)

df_results 
