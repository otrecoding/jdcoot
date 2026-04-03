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

import numpy as np
import pandas as pd
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import math

from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot

S_data = pd.read_csv("data.exp.csv")
T_data = pd.read_csv("data.met.csv")
#S_data = pd.read_csv("data_omique_S.csv")
#T_data = pd.read_csv("data_omique_T.csv")



# +
S_data = S_data.iloc[:, 2:]
T_data = T_data.iloc[:, 2:]

# renommer les colonnes
X_colsS = ['X'+str(i) for i in range(S_data.shape[1]-1)]
X_colsT = ['X'+str(i) for i in range(T_data.shape[1]-1)]

S_data.columns = X_colsS + ['Z']
T_data.columns = X_colsT + ['Z']

# convertir Z comme dans R
mapping = {
    "CIMP-low": 0,
    "CIMP-intermediate": 1,
    "CIMP-high": 2
}

S_data["Z"] = S_data["Z"].map(mapping)
T_data["Z"] = T_data["Z"].map(mapping)

S_data = S_data.dropna(subset=['Z'])
T_data = T_data.dropna(subset=['Z'])

# +
# identifiants des deux datasets
source_ids = S_data["X1"]
target_ids = T_data["X1"]

# trouver les individus dans target mais pas dans source
missing_in_source = target_ids[~target_ids.isin(source_ids)]

print("Individus dans target mais pas dans source :")
print(missing_in_source)
T_data = T_data[~T_data["X1"].isin(missing_in_source)]

# vérifier
print(T_data.shape)

# -

source.shape

target.shape

from jdcoot.utils import xcolumns
from sklearn.feature_selection import VarianceThreshold
source = S_data
target = T_data
x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values
#selector = VarianceThreshold(threshold=0.4)
selector = VarianceThreshold(threshold=0.1)
#x_source_reduced = selector.fit_transform(x_source)
x_source_reduced = selector.fit_transform(x_source)
selector = VarianceThreshold(threshold=0.1)
#selector = VarianceThreshold(threshold=0.1)
x_target_reduced = selector.fit_transform(x_target)
#x_target_reduced = selector.fit_transform(x_source)
x_target_reduced.shape
X_df = pd.DataFrame(x_source_reduced, columns=['X'+str(i) for i in range(x_source_reduced.shape[1])])
X_df['Z'] = source['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_reduced, columns=['X'+str(i) for i in range(x_target_reduced.shape[1])])
X_df['Z'] = target['Z'].values
target = X_df
source.shape, target.shape

# +
import seaborn as sns
import matplotlib.pyplot as plt
x_source_reduced2=x_source_reduced
corr_matrix = np.corrcoef(x_source_reduced.T, x_target_reduced.T)

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation between source and target features")
plt.show()

# +
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler_s = StandardScaler()
scaler_t = StandardScaler()

x_source_scaled = scaler_s.fit_transform(x_source_reduced)
x_target_scaled = scaler_t.fit_transform(x_target_reduced)


corr_matrix = np.corrcoef(
    x_source_scaled.T,
    x_target_scaled.T
)

p = x_source_scaled.shape[1]

corr_source_target = corr_matrix[:p, p:]

from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(-np.abs(corr_source_target))
x_target_scaled_aligned = x_target_scaled[:, col_ind]

signs = np.sign(corr_source_target[row_ind, col_ind])
x_target_scaled_aligned *= signs
#threshold = 0.6

#selected_cols = np.where(np.abs(corr_source_target)) > threshold)[0]
#selected_cols 
#x_source_selected = x_source_reduced[:, selected_cols]

corr_final = np.corrcoef(x_source_scaled.T, x_target_scaled_aligned.T)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(corr_final, cmap="coolwarm", center=0)
plt.title("Aligned correlation matrix")
plt.show()

X_df = pd.DataFrame(x_source_scaled, columns=['X'+str(i) for i in range(x_source_scaled.shape[1])])
X_df['Z'] = source['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_scaled_aligned, columns=['X'+str(i) for i in range(x_target_scaled_aligned.shape[1])])
X_df['Z'] = target['Z'].values
target = X_df
# -

target.shape

# +
from sklearn.model_selection import train_test_split

indices = np.arange(len(source))

# split stratifié sur Z
idx_source, idx_target = train_test_split(
    indices,
    test_size=0.5,
    stratify=source['Z'],
    random_state=42
)

# séparation
source_split = source.iloc[idx_source].reset_index(drop=True)
target_split = target.iloc[idx_target].reset_index(drop=True)

# +
source = source_split
target = target_split

source.shape, target.shape

# +
print("Source:")
print(source_split["Z"].value_counts().sort_index())

print("\nTarget:")
print(target_split["Z"].value_counts().sort_index())
# -

target

# +
a = np.random.choice(np.arange(len(source)), math.ceil(0.9 * len(source)), replace=False)
b = np.random.choice(np.arange(len(target)), math.ceil(0.9 * len(target)), replace=False)

S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
S = source.iloc[a, :].reset_index(drop=True)
T = target.iloc[b, :].reset_index(drop=True)
import os
import numpy as np
import tensorflow as tf


# -

S.shape;T.shape



# +
#Test predicteur

import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.model_selection import train_test_split
from jdcoot.utils import xcolumns, discrete_classifiers, discrete_accuracy


source_levels = np.sort(np.unique(S.Z))
target_levels = np.sort(np.unique(T.Z))

nClass = len(np.union1d(source_levels, target_levels))
categories = [np.arange(nClass)]

enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)

x_source_train = S.loc[:, xcolumns(S)].values
z_source_train = enc.fit_transform(S.Z.values[:, np.newaxis])

x_target_train = T.loc[:, xcolumns(T)].values
z_target_train = enc.fit_transform(T.Z.values[:, np.newaxis])

x_source_test = S_test.loc[:, xcolumns(S_test)].values
z_source_test = S_test.Z.values

x_target_test = T_test.loc[:, xcolumns(T_test)].values
z_target_test = T_test.Z.values

clf_source, clf_target = discrete_classifiers(S, T, "relu", "softmax")


clf_target.fit(x_target_train, z_target_train, batch_size=len(x_target_train), epochs=10, verbose=0, shuffle=False)

clf_source.fit(x_source_train, z_source_train, batch_size=len(x_source_train), epochs=10, verbose=0, shuffle=False)

z_target_pred = enc.inverse_transform(
        clf_target.predict(x_target_test, verbose=0)
    ).ravel()
z_source_pred = enc.inverse_transform(
        clf_source.predict(x_source_test, verbose=0)
    ).ravel()

perf_test_source = discrete_accuracy(z_source_pred,S_test.Z)
perf_test_target = discrete_accuracy(z_target_pred, T_test.Z)

# -

perf_test_source

perf_test_target

# +
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder as onehot
from jdcoot.utils import xcolumns, discrete_classifiers, discrete_accuracy

n_runs = 100

perf_source_list = []
perf_target_list = []

for i in range(n_runs):

    a = np.random.choice(np.arange(len(source)), math.ceil(0.9 * len(source)), replace=False)
    b = np.random.choice(np.arange(len(target)), math.ceil(0.9 * len(target)), replace=False)

    S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
    T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)

    S = source.iloc[a, :].reset_index(drop=True)
    T = target.iloc[b, :].reset_index(drop=True)

    source_levels = np.sort(np.unique(S.Z))
    target_levels = np.sort(np.unique(T.Z))

    nClass = len(np.union1d(source_levels, target_levels))
    categories = [np.arange(nClass)]

    enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)

    x_source_train = S.loc[:, xcolumns(S)].values
    z_source_train = enc.fit_transform(S.Z.values[:, np.newaxis])

    x_target_train = T.loc[:, xcolumns(T)].values
    z_target_train = enc.fit_transform(T.Z.values[:, np.newaxis])

    x_source_test = S_test.loc[:, xcolumns(S_test)].values
    z_source_test = S_test.Z.values

    x_target_test = T_test.loc[:, xcolumns(T_test)].values
    z_target_test = T_test.Z.values

    clf_source, clf_target = discrete_classifiers(S, T, "relu", "softmax")

    clf_target.fit(x_target_train, z_target_train,
                   batch_size=len(x_target_train), epochs=10, verbose=0, shuffle=False)

    clf_source.fit(x_source_train, z_source_train,
                   batch_size=len(x_source_train), epochs=10, verbose=0, shuffle=False)

    z_target_pred = enc.inverse_transform(
        clf_target.predict(x_target_test, verbose=0)
    ).ravel()

    z_source_pred = enc.inverse_transform(
        clf_source.predict(x_source_test, verbose=0)
    ).ravel()

    perf_test_source = discrete_accuracy(z_source_pred, z_source_test)
    perf_test_target = discrete_accuracy(z_target_pred, z_target_test)

    perf_source_list.append(perf_test_source)
    perf_target_list.append(perf_test_target)


mean_source = np.mean(perf_source_list)
mean_target = np.mean(perf_target_list)

print("Mean source test accuracy:", mean_source)
print("Mean target test accuracy:", mean_target)
# -

results = []
numRepetitions = 1
#algo = "sinkhorn"
algo = "emd"
reg = 0.1

T.shape

# +
#Test coot
algo = "sinkhorn"
reg = 0.01
S=source
T=target
S_test = source
T_test = target
pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(S, T, S_test, T_test,algo=algo,reg=reg,batch_size=20)
test_target


# -

pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(T,S, T_test, S_test,algo=algo,reg=reg,batch_size=20)
test_target

pure_target

# +
import numpy as np
from sklearn.preprocessing import OneHotEncoder as onehot

from jdcoot.coot import cot_numpy
from jdcoot.utils import xcolumns, discrete_accuracy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
source_levels = np.unique(source.Z)
target_levels = np.unique(source.Z)
size_target = target.Z.size

x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values

x_source = scaler.fit_transform(source.loc[:, xcolumns(source)].values)
x_target = scaler.fit_transform(target.loc[:, xcolumns(target)].values)

z_source = source.Z.values
z_target = target.Z.values
nClass = len(np.union1d(source_levels, target_levels))
categories = [np.arange(nClass)]

encoder = onehot(
        handle_unknown="ignore", sparse_output=False, categories=categories
    )

def one_hot(z):
        return encoder.fit_transform(z.reshape(-1, 1))

def one_cold(z):
        return encoder.inverse_transform(z).ravel()

Ts, Tv, cost = cot_numpy(
        X1=x_source,
        X2=x_target,
        niter=100,
        algo="emd",
        reg=reg,  
        algo2="emd",
        verbose=False,
    )

z_target_pred = size_target * np.dot(Ts.T, one_hot(z_source))


# -

x_target_pred = size_target * np.dot(Ts.T, x_source)


from sklearn.metrics import pairwise_distances
D = pairwise_distances(x_source, x_target_pred,metric="euclidean")

Tv

D[1,]

cols = np.where(Ts.T[1, :] > 0)[0]
cols
D[1,57]


z_target_pred 

one_cold(z_target_pred)

z_source

z_target

perf_pure_source = 1.0
perf_pure_target = discrete_accuracy(z_target, one_cold(z_target_pred))
perf_pure_target

algo = "sinkhorn"
reg = 0.1
pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(T, S, T_test, S_test,algo=algo,reg=reg,batch_size=len(x_source_train))
test_target


alpha= 0.5# hyperparamètre devant la loss a été optimé
algo = "emd"
reg = 0.1
pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S_test, T_test,algo=algo,reg=reg,batch_size=len(x_source_train), alpha=alpha)
test_target

# +
alpha=1.5
algo = "emd"
reg = 0.1

pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(T, S, T_test, S_test, algo=algo, reg=reg,batch_size=len(x_source_train), alpha=alpha)
test_target

# +
S = source
T = target


#Test predicteur

tf.config.experimental.enable_op_determinism()
source_levels = np.sort(np.unique(S.Z))
target_levels = np.sort(np.unique(T.Z))

nClass = len(np.union1d(source_levels, target_levels))
categories = [np.arange(nClass)]

enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)

x_source_train = S.loc[:, xcolumns(S)].values
z_source_train = enc.fit_transform(S.Z.values[:, np.newaxis])

x_target_train = T.loc[:, xcolumns(T)].values
z_target_train = enc.fit_transform(T.Z.values[:, np.newaxis])

x_source_test = S.loc[:, xcolumns(S)].values
z_source_test = S.Z.values

x_target_test = T.loc[:, xcolumns(T)].values
z_target_test = T.Z.values

clf_source, clf_target = discrete_classifiers(S, T, "relu", "softmax")


clf_target.fit(x_target_train, z_target_train, batch_size=len(x_target_train), epochs=10, verbose=0, shuffle=False)

clf_source.fit(x_source_train, z_source_train, batch_size=len(x_source_train), epochs=10, verbose=0, shuffle=False)

z_target_pred = enc.inverse_transform(
        clf_target.predict(x_target_test, verbose=0)
    ).ravel()
z_source_pred = enc.inverse_transform(
        clf_source.predict(x_source_test, verbose=0)
    ).ravel()

perf_test_source = discrete_accuracy(z_source_pred,S.Z)
perf_test_target = discrete_accuracy(z_target_pred, T.Z)

# -

perf_test_target

algo = "emd"
reg = 0.1
pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(S, T, S_test, T_test,algo=algo,reg=reg,batch_size=len(x_source_train))
test_target


pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(T, S, T, S,algo=algo,reg=reg,batch_size=len(x_source_train))
test_target

# +
alpha= 1.5



pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S, T, algo=algo, reg=reg,batch_size=len(x_source_train), alpha=alpha)
test_target

# +
alpha= 1.5

pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(T, S, T, S, algo=algo, reg=reg,batch_size=len(x_source_train), alpha=alpha)
test_target
# -

np.linspace(0, 2, 21)

# +
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import math

alpha_values = np.linspace(0, 2, 21)
num_folds = 5

kf_source = KFold(n_splits=num_folds, shuffle=True, random_state=42)
kf_target = KFold(n_splits=num_folds, shuffle=True, random_state=42)

results = []

algo = "emd"
reg = 0.1

best_alpha_global = None
best_score_global = -np.inf

for alpha in alpha_values:
    
    fold_scores = []
    
    for (train_idx_s, test_idx_s), (train_idx_t, test_idx_t) in zip(
        kf_source.split(source), 
        kf_target.split(target)
    ):
        
        S = source.iloc[train_idx_s].reset_index(drop=True)
        S_test = source.iloc[test_idx_s].reset_index(drop=True)

        T = target.iloc[train_idx_t].reset_index(drop=True)
        T_test = target.iloc[test_idx_t].reset_index(drop=True)

        pure_source, pure_target, test_source, test_target = \
            discrete_unsupervised_jdcoot(
                S, T,
                S_test, T_test,
                algo=algo,
                reg=reg,
                batch_size=len(S),
                alpha=alpha
            )

        score = test_target   

        fold_scores.append(score)

    mean_score = np.mean(fold_scores)

    if mean_score > best_score_global:
        best_score_global = mean_score
        best_alpha_global = alpha

    results.append({
        "alpha": alpha,
        "mean_score": mean_score
    })

df_results = pd.DataFrame(results)

print("Best alpha:", best_alpha_global)
print("Best score:", best_score_global)

df_results.to_excel("results_cv.xlsx", index=False)
# -

source.shape, target.shape

# +
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

numRepetitions = 100
num_folds = 5

algo = "sinkhorn"
reg = 0.1
alpha = 0.7   # fixe ici si tu ne fais pas tuning

results = []

for repe in range(numRepetitions):
    print("num repe :", repe + 1)

    kf_source = KFold(n_splits=num_folds, shuffle=True, random_state=repe)
    kf_target = KFold(n_splits=num_folds, shuffle=True, random_state=repe)

    for fold, ((train_s, test_s), (train_t, test_t)) in enumerate(
        zip(kf_source.split(source), kf_target.split(target))
    ):

        S = source.iloc[train_s].reset_index(drop=True)
        S_test = source.iloc[test_s].reset_index(drop=True)

        T = target.iloc[train_t].reset_index(drop=True)
        T_test = target.iloc[test_t].reset_index(drop=True)

        # ==========================
        # COOT
        # ==========================
        pure_source, pure_target, test_source, test_target = \
            discrete_unsupervised_coot(
                S, T, S_test, T_test,
                algo=algo,
                reg=reg,
                batch_size=len(S)
            )

        score_coot = test_target  

        results.append({
            "repetition": repe,
            "fold": fold,
            "recoding": "coot",
            "learning": "unsupervised",
            "score": score_coot,
        })

        # ==========================
        # JDCOOT
        # ==========================
        pure_source, pure_target, test_source, test_target = \
            discrete_unsupervised_jdcoot(
                S, T, S_test, T_test,
                algo=algo,
                reg=reg,
                batch_size=len(S),
                alpha=alpha
            )

        score_jdcoot = test_target  

        results.append({
            "repetition": repe,
            "fold": fold,
            "recoding": "jdcoot",
            "learning": "unsupervised",
            "score": score_jdcoot,
        })

# ==============================
# Résumé
# ==============================

df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(["recoding"])
    .agg(mean_score=("score", "mean"),
         std_score=("score", "std"))
    .reset_index()
)

print(df_summary)

df_summary.to_excel("results_cv_mean.xlsx", index=False)
