import sklearn
import scipy 
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
import math
from scipy.io import loadmat
from jdcoot.utils import xcolumns
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as onehot
from jdcoot.coot import cot_numpy
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_partial_jdcoot2 import discrete_partial_jdcoot2
from jdcoot.models.discrete_partial_jdcoot3 import discrete_partial_jdcoot3
from jdcoot.models.discrete_partial_jdcoot4 import discrete_partial_jdcoot4
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_partial_coot2 import discrete_partial_coot2
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference
from jdcoot.utils import discrete_classifier, xcolumns, discrete_accuracy
from jdcoot.utils import xcolumns, discrete_classifiers, discrete_accuracy
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import random
random.seed(42)

S_data = pd.read_csv("data_exp.csv")
T_data = pd.read_csv("data_met.csv")

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

S_data = S_data.iloc[:, 2:]
T_data = T_data.iloc[:, 2:]

X_colsS = ['X'+str(i) for i in range(S_data.shape[1]-1)]
X_colsT = ['X'+str(i) for i in range(T_data.shape[1]-1)]
S_data.columns = X_colsS + ['Z']
T_data.columns = X_colsT + ['Z']

source = S_data
target = T_data

source = source.dropna(subset=['Z'])
target = target.dropna(subset=['Z'])
x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values
selector = VarianceThreshold(threshold=10000)
x_source_reduced = selector.fit_transform(x_source)
selector = VarianceThreshold(threshold=0.1)#0.12
x_target_reduced = selector.fit_transform(x_target)
#x_target_reduced = selector.fit_transform(x_source)
x_target_reduced.shape
scaler_s = StandardScaler()
scaler_t = StandardScaler()

x_source_scaled = scaler_s.fit_transform(x_source_reduced)
x_target_scaled = scaler_t.fit_transform(x_target_reduced)
X_df = pd.DataFrame(x_source_scaled, columns=['X'+str(i) for i in range(x_source_reduced.shape[1])])
X_df['Z'] = source['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_scaled, columns=['X'+str(i) for i in range(x_target_reduced.shape[1])])
X_df['Z'] = target['Z'].values
target = X_df

corr_matrix = np.corrcoef(
    x_source_scaled.T,
    x_target_scaled.T
)

p = x_source_scaled.shape[1]

corr_source_target = corr_matrix[:p, p:]

# meilleure correspondance target pour chaque variable source
best_target = np.argmax(np.abs(corr_source_target), axis=1)

# signe des corrélations
signs = np.sign(corr_source_target[np.arange(p), best_target])
signs = np.where(signs == 0, 1, signs)

# inversion des colonnes source si corrélation négative
x_source_scaled = x_source_scaled * signs

#selected_cols = np.where(np.abs(corr_source_target)) > threshold)[0]
#selected_cols 
#x_source_selected = x_source_reduced[:, selected_cols]

corr_final = np.corrcoef(x_source_scaled.T, x_target_scaled.T)

X_df = pd.DataFrame(x_source_scaled, columns=['X'+str(i) for i in range(x_source_scaled.shape[1])])
X_df['Z'] = S_data['Z'].values
source= X_df
X_df = pd.DataFrame(x_target_scaled, columns=['X'+str(i) for i in range(x_target_scaled.shape[1])])
X_df['Z'] = T_data['Z'].values
target = X_df


n_reps = 10
algo = "emd"
reg = 0.1
alpha1 = 0.6
alpha2 = 0.4

results = []

indices = np.arange(len(source))

for rep in range(n_reps):

    print(f"\n=== Répétition {rep+1} ===")

    idx_source, idx_target = train_test_split(
        indices,
        test_size=0.5,
        stratify=source['Z'],
        random_state=rep
    )

    source2 = source.iloc[idx_source].reset_index(drop=True)
    target2 = target.iloc[idx_target].reset_index(drop=True)

    loo = LeaveOneOut()

    # ==========================
    # stockage des scores
    # ==========================

    scores_coot_source = []
    scores_coot_target = []
    scores_coot_mean = []

    scores_jdcoot_source = []
    scores_jdcoot_target = []
    scores_jdcoot_mean = []


    scores_reference_source = []
    scores_reference_target = []
    scores_reference_mean = []

    # ==========================
    # LOO
    # ==========================

    for train_idx, test_idx in loo.split(source2):

        S = source2.iloc[train_idx].reset_index(drop=True)
        S_test = source2.iloc[test_idx].reset_index(drop=True)

        T = target2.iloc[train_idx].reset_index(drop=True)
        T_test = target2.iloc[test_idx].reset_index(drop=True)

        l_source_train = np.where(np.isin(S["Z"], [0]))[0]
        l_source_test  = np.where(np.isin(S["Z"], [1, 2]))[0]

        l_target_train = np.where(np.isin(T["Z"], [1, 2]))[0]
        l_target_test  = np.where(np.isin(T["Z"], [0]))[0]

        # sécurité
        if (
            len(l_source_train) == 0
            or len(l_source_test) == 0
            or len(l_target_train) == 0
            or len(l_target_test) == 0
        ):
            continue

        # =========================================
        # COOT
        # =========================================

        _, _, test_source, test_target = discrete_partial_coot2(
            S,
            T,
            S_test,
            T_test,
            l_source_train,
            l_source_test,
            l_target_train,
            l_target_test,
            algo=algo,
            reg=reg,
            batch_size=len(S)
        )

        score_mean = (test_source + test_target) / 2

        scores_coot_source.append(test_source)
        scores_coot_target.append(test_target)
        scores_coot_mean.append(score_mean)

        # =========================================
        # JDCOOT
        # =========================================

        _, _, test_source, test_target = discrete_partial_jdcoot4(
            S,
            T,
            S_test,
            T_test,
            l_source_train,
            l_source_test,
            l_target_train,
            l_target_test,
            algo=algo,
            reg=reg,
            batch_size=len(S),
            alphaS=alpha1,
            alphaT=alpha2
        )

        score_mean = (test_source + test_target) / 2

        scores_jdcoot_source.append(test_source)
        scores_jdcoot_target.append(test_target)
        scores_jdcoot_mean.append(score_mean)

        pure_source, pure_target, test_source, test_target = discrete_partial_reference(
            S,
            T,
            S_test,
            T_test,
            l_source_train,
            l_source_test,
            l_target_train,
            l_target_test
        )

        score_mean = (test_source + test_target) / 2

        scores_reference_source.append(test_source)
        scores_reference_target.append(test_target)
        scores_reference_mean.append(score_mean)
    # =========================================
    # Résultats répétition
    # =========================================

    results.append({
        "rep": rep,
        "recoding": "coot",
        "source_score": np.mean(scores_coot_source),
        "target_score": np.mean(scores_coot_target),
        "mean_score": np.mean(scores_coot_mean)
    })

    results.append({
        "rep": rep,
        "recoding": "jdcoot",
        "source_score": np.mean(scores_jdcoot_source),
        "target_score": np.mean(scores_jdcoot_target),
        "mean_score": np.mean(scores_jdcoot_mean)
    })

    results.append({
        "rep": rep,
        "recoding": "reference",
        "source_score": np.mean(scores_reference_source),
        "target_score": np.mean(scores_reference_target),
        "mean_score": np.mean(scores_reference_mean)
    })
# =========================================
# DataFrame final
# =========================================

df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby("recoding")
    .agg(
        source_mean=("source_score", "mean"),
        source_std=("source_score", "std"),
        target_mean=("target_score", "mean"),
        target_std=("target_score", "std"),
        mean_mean=("mean_score", "mean"),
        mean_std=("mean_score", "std"),
    )
    .reset_index()
)

print(df_summary)
df_summary.to_excel("results_omiques_partial.xlsx", index=False)
