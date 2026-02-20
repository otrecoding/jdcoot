%load_ext autoreload
%autoreload 2

import sklearn
import scipy 
import numpy as np
import pandas as pd
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
import math
from scipy.io import loadmat

from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference


featuresToUse = ["CaffeNet4096", "GoogleNet1024"] 
#featuresToUse = ["CaffeNet4096", "CaffeNet4096"] 
sourceDomainName = ['amazon'] #['caltech10','amazon','webcam']
targetDomainName = ['amazon'] #['caltech10','amazon','webcam']

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

results = []
numRepetitions = 10
alpha =1.5# hyperparamètre devant la loss a été optimé
prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]

for repe in range(numRepetitions):
    print("num repe :", repe + 1)

    a = np.random.choice(np.arange(len(source)), math.ceil(0.8 * len(source)), replace=False)
    b = np.random.choice(np.arange(len(target)), math.ceil(0.8 * len(target)), replace=False)

    S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
    T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
    S = source.iloc[a, :].reset_index(drop=True)
    T = target.iloc[b, :].reset_index(drop=True)

    # =========================================================
    # UNSUPERVISED
    # =========================================================
     # COOT
    pure_source, pure_target, test_source, test_target = \
        discrete_unsupervised_coot(S, T, S_test, T_test,reg=1,alpha=alpha)

    results.append({
        "repetition": repe,
        "recoding": "coot",
        "learning": "unsupervised",
        "prop_source": 1,
        "prop_target": 0,
        "pure_source": pure_source,
        "test_source": test_source,
        "pure_target": pure_target,
        "test_target": test_target,
    })
    
    # JDCOOT
    pure_source, pure_target, test_source, test_target = \
        discrete_unsupervised_jdcoot(S, T, S_test, T_test,alpha=alpha)

    results.append({
        "repetition": repe,
        "recoding": "jdcoot",
        "learning": "unsupervised",
        "prop_source": 1,
        "prop_target": 0,
        "pure_source": pure_source,
        "test_source": test_source,
        "pure_target": pure_target,
        "test_target": test_target,
    })

   

df_results = pd.DataFrame(results)


df_results

df_summary = (
    df_results
    .groupby(
        ["recoding", "learning", "prop_source", "prop_target"],
        as_index=False
    )
    .agg(
        pure_source_mean=("pure_source", "mean"),
        pure_source_var=("pure_source", "var"),
        test_source_mean=("test_source", "mean"),
        test_source_var=("test_source", "var"),
        pure_target_mean=("pure_target", "mean"),
        pure_target_var=("pure_target", "var"),
        test_target_mean=("test_target", "mean"),
        test_target_var=("test_target", "var"),
    )
)

df_summary


 # =========================================================
    # SEMI-SUPERVISED
    # =========================================================
    for prop_target in prop_target_values_s:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_jdcoot(
                S, T, S_test, T_test,
                alpha=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = \
            discrete_semisupervised_coot(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_reference(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

    # =========================================================
    # PARTIAL
    # =========================================================
    for prop_target in prop_target_values_p:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(
                S, T, S_test, T_test,
                alpha=0.5,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })    

df_summary.to_excel("results_mean.xlsx", index=False)