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
import tensorflow as tf
from tensorflow.keras import backend as K

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import tensorflow as tf

tf.keras.backend.clear_session()

K.clear_session()
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
source = X_df
X_df = pd.DataFrame(x_target_scaled, columns=['X'+str(i) for i in range(x_target_scaled.shape[1])])
X_df['Z'] = T_data['Z'].values
target= X_df

alpha_values = np.linspace(0,1, 11)
algo = "emd"
reg = 0.1
n_reps = 10  # nombre de répétitions
results_all = []
indices = np.arange(len(source))
best_alpha_overall = None
best_score_overall = -np.inf

for rep in range(n_reps):
    print(f"\n=== Répétition {rep+1} ===")

    # ==========================
    # Split aléatoire individus
    # ==========================
    idx_source, idx_target = train_test_split(
        indices,
        test_size=0.5,
        stratify=source['Z'],
        random_state=rep
    )

    source2 = source.iloc[idx_source].reset_index(drop=True)
    target2 = target.iloc[idx_target].reset_index(drop=True)

    loo = LeaveOneOut()
    results_rep = []

    # ==========================
    # LOO interne
    # ==========================
    for train_idx, test_idx in loo.split(source2):
        S_train = source2.iloc[train_idx].reset_index(drop=True)
        S_test  = source2.iloc[test_idx].reset_index(drop=True)

        T_train = target2.iloc[train_idx].reset_index(drop=True)
        T_test  = target2.iloc[test_idx].reset_index(drop=True)

        for alpha in alpha_values:

            # ==========================
            # Appel fonction JDCOOT non supervisé
            # ==========================
            pure_source, pure_target, test_source, test_target = \
                discrete_unsupervised_jdcoot(
                    S_train,
                    T_train,
                    S_test,
                    T_test,
                    algo=algo,
                    reg=reg,
                    batch_size=10,
                    alpha=alpha
                )

            results_rep.append({
                "rep": rep+1,
                "alpha": alpha,
                "loo_score": test_target
            })

    # ==========================
    # Sauvegarde résultats par répétition
    # ==========================
    df_rep = pd.DataFrame(results_rep)
    results_all.append(df_rep)

    # ==========================
    # Meilleur alpha de cette rép
    # ==========================
    mean_scores = df_rep.groupby("alpha")["loo_score"].mean()
    best_alpha_rep = mean_scores.idxmax()
    best_score_rep = mean_scores.max()

    print(f"Best alpha (rep {rep+1}): {best_alpha_rep}")
    print(f"Best LOO score (rep {rep+1}): {best_score_rep}")

    if best_score_rep > best_score_overall:
        best_score_overall = best_score_rep
        best_alpha_overall = best_alpha_rep

# ==========================
# Résultats globaux
# ==========================
df_results_all = pd.concat(results_all, ignore_index=True)
df_results_all.to_excel("results_loo_all_reps.xlsx", index=False)

print("\n=== Résultat global ===")
print("Best alpha overall:", best_alpha_overall)
print("Best LOO score overall:", best_score_overall)

df_summary = df_results_all.groupby("alpha")["loo_score"].mean().reset_index()
df_summary.to_excel("results_loo_mean_alpha_StoT.xlsx", index=False)