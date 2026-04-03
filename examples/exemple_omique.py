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
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
from jdcoot.utils import xcolumns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as onehot
from jdcoot.coot import cot_numpy
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_partial_reference import discrete_partial_reference
from jdcoot.utils import discrete_accuracy
from jdcoot.utils import discrete_classifiers
from sklearn.model_selection import LeaveOneOut

# -

S_data = pd.read_csv("data_exp.csv")
T_data = pd.read_csv("data_met.csv")
#S_data = pd.read_csv("data_omique_S.csv")
#T_data = pd.read_csv("data_omique_T.csv")

# +
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

S_data = S_data.iloc[:, 2:]
T_data = T_data.iloc[:, 2:]

# +
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
source.shape,target.shape
# -

source.shape,target.shape

# +
source_levels = np.unique(source.Z)
target_levels = np.unique(source.Z)
size_target = target.Z.size

x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values

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
        reg=0.01,  
        algo2="emd",
        verbose=False,
    )

z_target_pred = size_target * np.dot(Ts.T, one_hot(z_source))
perf_pure_target = discrete_accuracy(z_target, one_cold(z_target_pred))
perf_pure_target



# +
# Mauvaise performance, je modifie les données pour avoir au maximum des corrélations positives (en changeant le signe quand la corrélation est négative)

# +
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
# -

plt.figure(figsize=(8,6))
sns.heatmap(corr_final, cmap="coolwarm", center=0)
plt.title("Final correlation between source and target features")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation between source and target features")
plt.show()

# +
X_df = pd.DataFrame(x_source_scaled, columns=['X'+str(i) for i in range(x_source_scaled.shape[1])])
X_df['Z'] = S_data['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_scaled, columns=['X'+str(i) for i in range(x_target_scaled.shape[1])])
X_df['Z'] = T_data['Z'].values
target = X_df

x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values

z_source = source.Z.values
z_target = target.Z.values

source_levels = np.unique(source.Z)
target_levels = np.unique(target.Z)

nClass = len(np.union1d(source_levels, target_levels))
categories = [np.arange(nClass)]

Ts, Tv, cost = cot_numpy(
        X1=x_source,
        X2=x_target,
        niter=100,
        algo="emd",
        reg=0.01,  
        algo2="emd",
        verbose=False,
    )

z_target_pred = size_target * np.dot(Ts.T, one_hot(z_source))
perf_pure_target = discrete_accuracy(z_target, one_cold(z_target_pred))
perf_pure_target

# +
# je cree des echantillons independants pour le train et le test, en prenant un individu de chaque classe pour le train et le reste pour le test

# +
X_df = pd.DataFrame(x_source_scaled, columns=['X'+str(i) for i in range(x_source_scaled.shape[1])])
X_df['Z'] = source['Z'].values
source = X_df
X_df = pd.DataFrame(x_target_scaled, columns=['X'+str(i) for i in range(x_target_scaled.shape[1])])
X_df['Z'] = target['Z'].values
target = X_df
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
source2 = source_split
target2 = target_split

source2.shape, target2.shape

# +
x_source = source2.loc[:, xcolumns(source)].values
x_target = target2.loc[:, xcolumns(target)].values

z_source = source2.Z.values
z_target = target2.Z.values
nClass = len(np.union1d(source_levels, target_levels))
categories = [np.arange(nClass)]

Ts, Tv, cost = cot_numpy(
        X1=x_source,
        X2=x_target,
        niter=100,
        algo="emd",
        reg=0.01,  
        algo2="emd",
        verbose=False,
    )

z_target_pred = size_target * np.dot(Ts.T, one_hot(z_source))
perf_pure_target = discrete_accuracy(z_target, one_cold(z_target_pred))
perf_pure_target
# -

algo = "emd"
reg = 0.1
S=source2
T=target2
S_test = source2
T_test = target2
pure_source, pure_target, test_source, test_target =  discrete_unsupervised_coot(S, T, S_test, T_test,algo=algo,reg=reg,batch_size=20)
test_target

# +
alpha=1.2
algo = "emd"
reg = 0.1
alpha = 1
pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S_test, T_test, algo=algo, reg=reg,batch_size=20, alpha=alpha)
test_target

#JDCOOT = COOT car il converge à une itération !!
# -

from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_partial_coot2 import discrete_partial_coot2

# +
S=source2
T=target2
S_test = source2
T_test = target2
l_source_train = np.where(np.isin(S['Z'], [0]))[0]
l_source_test  = np.where(np.isin(S['Z'], [1,2]))[0]

# ===== TARGET =====
l_target_train = np.where(np.isin(T['Z'], [1,2]))[0]
l_target_test  = np.where(np.isin(T['Z'], [0]))[0]
# -

pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="emd",reg=0.1,batch_size=len(S)            
            )
test_target

test_source

pure_source, pure_target, test_source, test_target = discrete_partial_coot2(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="emd",reg=0.1,batch_size=len(S)            
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test
            )
test_target

test_source

# +
prop_source=0.5
prop_target=0.1
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

pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="emd",reg=0.1,batch_size=len(S)            
            )
test_target

test_source

pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test
            )
test_target

test_source

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
# -

pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="emd",reg=0.1,batch_size=len(S)            
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_coot2(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test,algo="emd",reg=0.1,batch_size=len(S)            
            )
test_target

pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,l_source_train, l_source_test,l_target_train, l_target_test
            )
test_target

# +
perf_source_all = []
perf_target_all = []
n_reps = 5 
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

    perf_source_list = []
    perf_target_list = []

    # ==========================
    # LOO interne
    # ==========================
    for train_idx, test_idx in loo.split(source2):

        # même split pour source et target (comme ton 1er code)
        S = source2.iloc[train_idx].reset_index(drop=True)
        S_test = source2.iloc[test_idx].reset_index(drop=True)

        T = target2.iloc[train_idx].reset_index(drop=True)
        T_test = target2.iloc[test_idx].reset_index(drop=True)

        # ==========================
        # Encodage
        # ==========================
        source_levels = np.sort(np.unique(S.Z))
        target_levels = np.sort(np.unique(T.Z))

        nClass = len(np.union1d(source_levels, target_levels))
        categories = [np.arange(nClass)]

        enc = onehot(handle_unknown="ignore", sparse_output=False, categories=categories)

        # train
        x_source_train = S.loc[:, xcolumns(S)].values
        z_source_train = enc.fit_transform(S.Z.values[:, np.newaxis])

        x_target_train = T.loc[:, xcolumns(T)].values
        z_target_train = enc.fit_transform(T.Z.values[:, np.newaxis])

        # test
        x_source_test = S_test.loc[:, xcolumns(S_test)].values
        z_source_test = S_test.Z.values

        x_target_test = T_test.loc[:, xcolumns(T_test)].values
        z_target_test = T_test.Z.values

        # ==========================
        # Modèles
        # ==========================
        clf_source, clf_target = discrete_classifiers(S, T, "relu", "softmax")

        clf_target.fit(
            x_target_train, z_target_train,
            batch_size=len(x_target_train), epochs=10, verbose=0, shuffle=False
        )

        clf_source.fit(
            x_source_train, z_source_train,
            batch_size=len(x_source_train), epochs=10, verbose=0, shuffle=False
        )

        # ==========================
        # Prédictions
        # ==========================
        z_target_pred = enc.inverse_transform(
            clf_target.predict(x_target_test, verbose=0)
        ).ravel()

        z_source_pred = enc.inverse_transform(
            clf_source.predict(x_source_test, verbose=0)
        ).ravel()

        # ==========================
        # Scores
        # ==========================
        perf_source_list.append(discrete_accuracy(z_source_pred, z_source_test))
        perf_target_list.append(discrete_accuracy(z_target_pred, z_target_test))

    # ==========================
    # Moyenne par répétition
    # ==========================
    perf_source_all.append(np.mean(perf_source_list))
    perf_target_all.append(np.mean(perf_target_list))


# ==========================
# Résultat final
# ==========================
print("Source accuracy:", np.mean(perf_source_all))
print("Target accuracy:", np.mean(perf_target_all))

# +
alpha_values = np.linspace(0, 2, 5)
algo = "emd"
reg = 0.1
n_reps = 5  # nombre de répétitions
results_all = []

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
                    batch_size=len(S_train),
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

# +
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut

algo = "emd"
reg = 0.1
alpha = 0.01

n_reps = 50  # nombre de répétitions Monte Carlo
results = []

indices = np.arange(len(source))

for rep in range(n_reps):

    print(f"\n=== Répétition {rep+1} ===")

    # ==========================
    # Split aléatoire individus
    # ==========================
    idx_source, idx_target = train_test_split(
        indices,
        test_size=0.5,
        stratify=source['Z'],
        random_state=rep  # change à chaque répétition
    )

    source2 = source.iloc[idx_source].reset_index(drop=True)
    target2 = target.iloc[idx_target].reset_index(drop=True)

    loo = LeaveOneOut()

    scores_coot = []
    scores_jdcoot = []

    # ==========================
    # LOO interne
    # ==========================
    for train_idx, test_idx in loo.split(source2):

        S = target2.iloc[train_idx].reset_index(drop=True)
        S_test = target2.iloc[test_idx].reset_index(drop=True)

        T = source2.iloc[train_idx].reset_index(drop=True)
        T_test = source2.iloc[test_idx].reset_index(drop=True)

        # COOT
        _, _, _, test_target = discrete_unsupervised_coot(
            S, T, S_test, T_test,
            algo=algo,
            reg=reg,
            batch_size=len(S)
        )
        scores_coot.append(test_target)

        # JDCOOT
        _, _, _, test_target = discrete_unsupervised_jdcoot(
            S, T, S_test, T_test,
            algo=algo,
            reg=reg,
            batch_size=len(S),
            alpha=alpha
        )
        scores_jdcoot.append(test_target)

    # ==========================
    # Moyenne sur LOO
    # ==========================
    results.append({
        "rep": rep,
        "recoding": "coot",
        "mean_score": np.mean(scores_coot)
    })

    results.append({
        "rep": rep,
        "recoding": "jdcoot",
        "mean_score": np.mean(scores_jdcoot)
    })

df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(["recoding"])
    .agg(mean_score=("score", "mean"),
         std_score=("score", "std"))
    .reset_index()
)

print(df_summary)

df_summary.to_excel("results_loo_mean.xlsx", index=False)
