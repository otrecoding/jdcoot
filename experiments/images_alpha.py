import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot

featuresToUse = ["CaffeNet4096", "GoogleNet1024"]

# =========================================================
# PARAMETERS
# =========================================================

# Identical-domain pairs only: A->A, C->C, W->W
domainPairs = [
    ('amazon', 'amazon'),
    ('caltech10', 'caltech10'),
    ('webcam', 'webcam'),
]

n_splits = 5     # Number of folds K
algo = "sinkhorn"
reg = 1
batch_size = 20

# Search grid for alpha
alpha_values = [
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.5, 3.0, 5.0
]

results = []


def stratified_kfold_splits(df, class_col='Z', n_splits=5, seed=None):
    """
    Generate stratified folds with the same class proportions in each fold.

    Return a list of (train_pool, test) tuples, one per fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df[class_col].values

    splits = []
    for train_idx, test_idx in skf.split(df, y):
        train_pool = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        splits.append((train_pool, test))

    return splits


# =========================================================
# LOOP OVER SAME-DOMAIN PAIRS ONLY
# =========================================================

for source_name, target_name in domainPairs:

    print("=" * 60)
    print(f"SOURCE = {source_name} | TARGET = {target_name}")
    print("=" * 60)

    # === LOAD SOURCE ===
    possible_data = loadmat(
        os.path.join("data/", "features", featuresToUse[0], source_name + '.mat')
    )
    feat = possible_data['fts'].astype(float)
    labels = possible_data['labels'].ravel()
    S_data = [feat, labels]

    # === LOAD TARGET ===
    possible_data = loadmat(
        os.path.join("data/", "features", featuresToUse[1], target_name + '.mat')
    )
    feat = possible_data['fts'].astype(float)
    labels = possible_data['labels'].ravel()
    T_data = [feat, labels]

    # === DATAFRAMES ===
    source = pd.DataFrame(
        np.concatenate((S_data[0], S_data[1].reshape(-1, 1)), axis=1),
        columns=['X' + str(i) for i in range(S_data[0].shape[1])] + ['Z']
    )
    target = pd.DataFrame(
        np.concatenate((T_data[0], T_data[1].reshape(-1, 1)), axis=1),
        columns=['X' + str(i) for i in range(T_data[0].shape[1])] + ['Z']
    )

    source.loc[:, 'Z'] = source.loc[:, 'Z'] - 1
    target.loc[:, 'Z'] = target.loc[:, 'Z'] - 1

    # =========================================================
    # STRATIFIED K-FOLD
    # =========================================================

    source_folds = stratified_kfold_splits(source, class_col='Z', n_splits=n_splits)
    target_folds = stratified_kfold_splits(target, class_col='Z', n_splits=n_splits)

    # =====================================================
    # LOOP OVER FOLDS
    # =====================================================

    for fold_idx in range(n_splits):

        print(f"  Fold {fold_idx + 1}/{n_splits}")

        S, S_test = source_folds[fold_idx]
        T, T_test = target_folds[fold_idx]

        # =================================================
        # HYPERPARAMETER SEARCH ON ALPHA (JDCOOT UNSUPERVISED)
        # =================================================

        for alpha_val in alpha_values:

            pure_source, pure_target, test_source, test_target = \
                discrete_unsupervised_jdcoot(
                    S, T, S_test, T_test,
                    algo=algo,
                    reg=reg,
                    alpha=alpha_val,
                    batch_size=batch_size
                )

            results.append({
                "source": source_name,
                "target": target_name,
                "fold": fold_idx,
                "recoding": "jdcoot",
                "learning": "unsupervised",
                "alpha": alpha_val,
                "pure_source": pure_source,
                "test_source": test_source,
                "pure_target": pure_target,
                "test_target": test_target,
            })

# =========================================================
# RESULTS
# =========================================================

df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(
        ["source", "target", "recoding", "learning", "alpha"],
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

df_summary.to_excel("results_alpha_search_same_domain.xlsx", index=False)

print(df_summary)

# =========================================================
# BEST ALPHA PER DOMAIN (based on test_target_mean; adjust if needed)
# =========================================================

best_alpha = (
    df_summary
    .sort_values("test_target_mean", ascending=False)
    .groupby("source", as_index=False)
    .first()
)

print("\nMeilleur alpha par domaine (selon test_target_mean) :")
print(best_alpha[["source", "target", "alpha", "test_target_mean"]])