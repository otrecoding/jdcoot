import os
import sys
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut

from jdcoot.utils import xcolumns
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot

random.seed(42)

# =========================================================
# GPU
# =========================================================



gpus = tf.config.list_physical_devices('GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

K.clear_session()

# =========================================================
# Load data
# =========================================================

S_data = pd.read_csv("data_exp.csv")
T_data = pd.read_csv("data_met.csv")

# =========================================================
# Rename columns
# =========================================================

X_colsS = ['X' + str(i) for i in range(S_data.shape[1] - 1)]
X_colsT = ['X' + str(i) for i in range(T_data.shape[1] - 1)]

S_data.columns = X_colsS + ['Z']
T_data.columns = X_colsT + ['Z']

# =========================================================
# Map labels
# =========================================================

mapping = {
    "CIMP-low": 0,
    "CIMP-intermediate": 1,
    "CIMP-high": 2
}

S_data["Z"] = S_data["Z"].map(mapping)
T_data["Z"] = T_data["Z"].map(mapping)

S_data = S_data.dropna(subset=['Z'])
T_data = T_data.dropna(subset=['Z'])

# =========================================================
# Synchronize individuals
# =========================================================

source_ids = S_data["X1"]
target_ids = T_data["X1"]

missing_in_source = target_ids[~target_ids.isin(source_ids)]

print("Individus dans target mais pas dans source :")
print(missing_in_source)

T_data = T_data[~T_data["X1"].isin(missing_in_source)]

print("Shape target après filtrage :", T_data.shape)

# =========================================================
# Remove IDs
# =========================================================

S_data = S_data.iloc[:, 2:]
T_data = T_data.iloc[:, 2:]

X_colsS = ['X' + str(i) for i in range(S_data.shape[1] - 1)]
X_colsT = ['X' + str(i) for i in range(T_data.shape[1] - 1)]

S_data.columns = X_colsS + ['Z']
T_data.columns = X_colsT + ['Z']

source = S_data
target = T_data

# =========================================================
# Matrices X
# =========================================================

x_source = source.loc[:, xcolumns(source)].values
x_target = target.loc[:, xcolumns(target)].values

# =========================================================
# Variance threshold
# =========================================================

selector = VarianceThreshold(threshold=10000)
x_source_reduced = selector.fit_transform(x_source)

selector = VarianceThreshold(threshold=0.1)
x_target_reduced = selector.fit_transform(x_target)

print("Shape source :", x_source_reduced.shape)
print("Shape target :", x_target_reduced.shape)

# =========================================================
# Standardization
# =========================================================

scaler_s = StandardScaler()
scaler_t = StandardScaler()

x_source_scaled = scaler_s.fit_transform(x_source_reduced)
x_target_scaled = scaler_t.fit_transform(x_target_reduced)

# =========================================================
# Correlations
# =========================================================

corr_matrix = np.corrcoef(
    x_source_scaled.T,
    x_target_scaled.T
)

p = x_source_scaled.shape[1]

corr_source_target = corr_matrix[:p, p:]

best_target = np.argmax(np.abs(corr_source_target), axis=1)

signs = np.sign(
    corr_source_target[np.arange(p), best_target]
)

signs = np.where(signs == 0, 1, signs)

x_source_scaled = x_source_scaled * signs

# =========================================================
# Reconstruct dataframes
# =========================================================

X_df = pd.DataFrame(
    x_source_scaled,
    columns=['X' + str(i) for i in range(x_source_scaled.shape[1])]
)

X_df['Z'] = S_data['Z'].values

source = X_df

X_df = pd.DataFrame(
    x_target_scaled,
    columns=['X' + str(i) for i in range(x_target_scaled.shape[1])]
)

X_df['Z'] = T_data['Z'].values

target = X_df

# =========================================================
# Parameters
# =========================================================

alpha_values = np.linspace(0, 1, 11)
algo = "emd"
reg = 0.1
n_reps = 10

results_all = []

best_alpha_source_overall = None
best_alpha_target_overall = None
best_alpha_mean_overall = None

best_score_source_overall = -np.inf
best_score_target_overall = -np.inf
best_score_mean_overall = -np.inf

indices = np.arange(len(source))

# =========================================================
# Live Excel file
# =========================================================

excel_file = "results_partial_jdcoot_live.xlsx"

if not os.path.exists(excel_file):

    pd.DataFrame(columns=[
        "rep",
        "alpha",
        "test_source",
        "test_target",
        "mean_score"
    ]).to_excel(excel_file, index=False)

# =========================================================
# Repetition loop
# =========================================================

for rep in range(n_reps):

    print("\n====================================")
    print(f"REP {rep + 1}")
    print("====================================")

    idx_source, idx_target = train_test_split(
        indices,
        test_size=0.5,
        stratify=source["Z"],
        random_state=rep
    )

    source2 = source.iloc[idx_source].reset_index(drop=True)
    target2 = target.iloc[idx_target].reset_index(drop=True)

    results_rep = []

    loo = LeaveOneOut()

    # =====================================================
    # Leave-one-out
    # =====================================================

    for train_idx, test_idx in loo.split(source2):

        S = source2.iloc[train_idx].reset_index(drop=True)
        T = target2.iloc[train_idx].reset_index(drop=True)

        S_test = source2.iloc[test_idx].reset_index(drop=True)
        T_test = target2.iloc[test_idx].reset_index(drop=True)

        # =================================================
        # Partial indices
        # =================================================

        l_source_train = np.where(
            np.isin(S["Z"], [0])
        )[0]

        l_source_test = np.where(
            np.isin(S["Z"], [1, 2])
        )[0]

        l_target_train = np.where(
            np.isin(T["Z"], [1, 2])
        )[0]

        l_target_test = np.where(
            np.isin(T["Z"], [0])
        )[0]

        # =================================================
        # Guard against empty label groups
        # =================================================

        if (
            len(l_source_train) == 0
            or len(l_source_test) == 0
            or len(l_target_train) == 0
            or len(l_target_test) == 0
        ):
            continue

        # =================================================
        # Alpha loop
        # =================================================

        for alpha in alpha_values:

            print(f"\nREP = {rep+1} | alpha = {alpha}")

            _, _, test_source, test_target = discrete_partial_jdcoot(
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
                alphaS=alpha,
                alphaT=alpha,
            )

            score_mean = (test_source + test_target) / 2

            print("test_source =", test_source)
            print("test_target =", test_target)
            print("mean_score  =", score_mean)

            # =============================================
            # Store in memory
            # =============================================

            new_row = {
                "rep": rep + 1,
                "alpha": alpha,
                "test_source": test_source,
                "test_target": test_target,
                "mean_score": score_mean
            }

            results_rep.append(new_row)

            # =============================================
            # Save to Excel immediately
            # =============================================

            df_new = pd.DataFrame([new_row])

            with pd.ExcelWriter(
                excel_file,
                mode="a",
                engine="openpyxl",
                if_sheet_exists="overlay"
            ) as writer:

                workbook = writer.book
                sheet = workbook["Sheet1"]

                startrow = sheet.max_row

                df_new.to_excel(
                    writer,
                    index=False,
                    header=False,
                    startrow=startrow
                )

    # =====================================================
    # Repetition dataframe
    # =====================================================

    df_rep = pd.DataFrame(results_rep)

    results_all.append(df_rep)

    # =====================================================
    # Best source alpha
    # =====================================================

    mean_source = (
        df_rep.groupby("alpha")["test_source"]
        .mean()
    )

    alpha_source = mean_source.idxmax()
    score_source = mean_source.max()

    # =====================================================
    # Best target alpha
    # =====================================================

    mean_target = (
        df_rep.groupby("alpha")["test_target"]
        .mean()
    )

    alpha_target = mean_target.idxmax()
    score_target = mean_target.max()

    # =====================================================
    # Best mean alpha
    # =====================================================

    mean_global = (
        df_rep.groupby("alpha")["mean_score"]
        .mean()
    )

    alpha_mean = mean_global.idxmax()
    score_mean = mean_global.max()

    print("\n----------- RESULTATS REP -----------")

    print("Best alpha source :", alpha_source)
    print("Best score source :", score_source)

    print("Best alpha target :", alpha_target)
    print("Best score target :", score_target)

    print("Best alpha mean   :", alpha_mean)
    print("Best score mean   :", score_mean)

    # =====================================================
    # Best overall source
    # =====================================================

    if score_source > best_score_source_overall:

        best_score_source_overall = score_source
        best_alpha_source_overall = alpha_source

    # =====================================================
    # Best overall target
    # =====================================================

    if score_target > best_score_target_overall:

        best_score_target_overall = score_target
        best_alpha_target_overall = alpha_target

    # =====================================================
    # Best overall mean
    # =====================================================

    if score_mean > best_score_mean_overall:

        best_score_mean_overall = score_mean
        best_alpha_mean_overall = alpha_mean

# =========================================================
# Final concatenation
# =========================================================

df_results_all = pd.concat(
    results_all,
    ignore_index=True
)

# =========================================================
# Save all results
# =========================================================

df_results_all.to_excel(
    "results_partial_jdcoot_all.xlsx",
    index=False
)

# =========================================================
# Summary
# =========================================================

summary = (
    df_results_all
    .groupby("alpha")[[
        "test_source",
        "test_target",
        "mean_score"
    ]]
    .mean()
    .reset_index()
)

summary.to_excel(
    "results_omiques_partial_jdcoot_alpha.xlsx",
    index=False
)

# =========================================================
# Final results
# =========================================================

print("\n====================================")
print("RESULTAT FINAL")
print("====================================")

print("Best alpha source overall =",
      best_alpha_source_overall)

print("Best score source overall =",
      best_score_source_overall)

print("Best alpha target overall =",
      best_alpha_target_overall)

print("Best score target overall =",
      best_score_target_overall)

print("Best alpha mean overall   =",
      best_alpha_mean_overall)

print("Best score mean overall   =",
      best_score_mean_overall)
