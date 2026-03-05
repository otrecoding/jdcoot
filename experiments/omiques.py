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


S_data = pd.read_csv("data_omique_S.csv")
T_data = pd.read_csv("data_omique_T.csv")
X_colsS = ['X'+str(i) for i in range(S_data.shape[1]-1)]
X_colsT = ['X'+str(i) for i in range(T_data.shape[1]-1)]
S_data.columns = X_colsS + ['Z']
T_data.columns = X_colsT + ['Z']

source = S_data
target = T_data

source = S_data
target = T_data
source = source.dropna(subset=['Z'])
target = target.dropna(subset=['Z'])
x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values
#selector = VarianceThreshold(threshold=0.5)
selector = VarianceThreshold(threshold=0.3)
x_source_reduced = selector.fit_transform(x_source)
selector = VarianceThreshold(threshold=0.12)
#selector = VarianceThreshold(threshold=0.1)
x_target_reduced = selector.fit_transform(x_target)
x_target_reduced.shape
X_df = pd.DataFrame(x_source_reduced, columns=['X'+str(i) for i in range(x_source_reduced.shape[1])])
X_df['Z'] = source['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_reduced, columns=['X'+str(i) for i in range(x_target_reduced.shape[1])])
X_df['Z'] = target['Z'].values
target = X_df


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import math

alpha_values = np.linspace(0, 2, 21)
num_folds = 5

kf_source = KFold(n_splits=num_folds, shuffle=True, random_state=42)
kf_target = KFold(n_splits=num_folds, shuffle=True, random_state=42)

results = []

algo = "sinkhorn"
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

df_results.to_excel("results_alpha.xlsx", index=False)


import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

numRepetitions = 10
num_folds = 5

algo = "emd"
reg = 0.1
alpha = 0.6   

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

        score_coot = test_target  # ⚠️ mets ta vraie métrique

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

        score_jdcoot = test_target  # ⚠️ mets ta vraie métrique

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