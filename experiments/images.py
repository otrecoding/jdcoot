import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference

featuresToUse = ["CaffeNet4096", "GoogleNet1024"]

# =========================================================
# PARAMETERS
# =========================================================

sourceDomainName = ['amazon', 'caltech10', 'webcam']
targetDomainName = ['caltech10', 'amazon', 'webcam']

n_splits = 5          # nombre de folds K (remplace numRepetitions)
algo = "sinkhorn"
reg = 1
batch_size = 20
alpha = 1
prop_values = [1, 2, 3, 4, 5]
n_per_class_train = 20

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


def sample_n_per_class(df, class_col='Z', n=20, seed=None):
    """Sample up to n observations per class."""
    rng = np.random.default_rng(seed)
    idx_list = []

    for z in np.unique(df[class_col]):
        idx = np.where(df[class_col] == z)[0]
        n_take = min(n, len(idx))
        chosen = rng.choice(idx, size=n_take, replace=False)
        idx_list.extend(chosen)

    return df.iloc[idx_list].reset_index(drop=True)


# =========================================================
# LOOP OVER ALL SOURCE / TARGET COMBINATIONS
# =========================================================

for source_name in sourceDomainName:

    for target_name in targetDomainName:

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


        source_folds = stratified_kfold_splits(source, class_col='Z', n_splits=n_splits)
        target_folds = stratified_kfold_splits(target, class_col='Z', n_splits=n_splits)


        for fold_idx in range(n_splits):

            print(f"Fold {fold_idx + 1}/{n_splits}")

            source_train_pool, S_test = source_folds[fold_idx]
            target_train_pool, T_test = target_folds[fold_idx]

            S = source_train_pool
            T = target_train_pool

            # =================================================
            # UNSUPERVISED
            # =================================================

            # COOT
            pure_source, pure_target, test_source, test_target = \
                discrete_unsupervised_coot(
                    S, T, S_test, T_test,
                    algo=algo,
                    reg=reg,
                    batch_size=batch_size
                )

            results.append({
                "source": source_name,
                "target": target_name,
                "fold": fold_idx,
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
                discrete_unsupervised_jdcoot(
                    S, T, S_test, T_test,
                    algo=algo,
                    reg=reg,
                    alpha=alpha,
                    batch_size=batch_size
                )

            results.append({
                "source": source_name,
                "target": target_name,
                "fold": fold_idx,
                "recoding": "jdcoot",
                "learning": "unsupervised",
                "prop_source": 1,
                "prop_target": 0,
                "pure_source": pure_source,
                "test_source": test_source,
                "pure_target": pure_target,
                "test_target": test_target,
            })

            # =================================================
            # SEMI-SUPERVISED + PARTIAL
            # =================================================

            for prop_values_s in prop_values:

                # TARGET LABELS
                l_target_train = []
                for z in np.unique(T['Z']):
                    idx = np.where(T['Z'] == z)[0]
                    chosen = np.random.choice(idx, size=prop_values_s, replace=False)
                    l_target_train.extend(chosen)

                l_target_train = np.array(l_target_train)
                l_target_test = np.setdiff1d(np.arange(len(T['Z'])), l_target_train)

                # SEMISUPERVISED COOT
                pure_source, pure_target, test_source, test_target = \
                    discrete_semisupervised_coot(
                        S, T, S_test, T_test,
                        l_target_train, l_target_test,
                        algo=algo, reg=reg, batch_size=batch_size, alpha=alpha
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "coot", "learning": "semisupervised",
                    "prop_source": 1, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

                # SEMISUPERVISED JDCOOT
                pure_source, pure_target, test_source, test_target = \
                    discrete_semisupervised_jdcoot(
                        S, T, S_test, T_test,
                        l_target_train, l_target_test,
                        algo=algo, reg=reg, batch_size=batch_size, alpha=alpha
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "jdcoot", "learning": "semisupervised",
                    "prop_source": 1, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

                # SEMISUPERVISED REFERENCE
                pure_source, pure_target, test_source, test_target = \
                    discrete_semisupervised_reference(
                        S, T, S_test, T_test,
                        l_target_train, l_target_test,
                        algo=algo, reg=reg, batch_size=batch_size
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "reference", "learning": "semisupervised",
                    "prop_source": 1, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

                # SOURCE LABELS
                l_source_train = []
                for z in np.unique(S['Z']):
                    idx = np.where(S['Z'] == z)[0]
                    chosen = np.random.choice(idx, size=prop_values_s, replace=False)
                    l_source_train.extend(chosen)

                l_source_train = np.array(l_source_train)
                l_source_test = np.setdiff1d(np.arange(len(S['Z'])), l_source_train)

                # PARTIAL COOT
                pure_source, pure_target, test_source, test_target = \
                    discrete_partial_coot(
                        S, T, S_test, T_test,
                        l_source_train, l_source_test,
                        l_target_train, l_target_test,
                        algo=algo, reg=reg, batch_size=batch_size
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "coot", "learning": "partial",
                    "prop_source": prop_values_s, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

                # PARTIAL JDCOOT
                pure_source, pure_target, test_source, test_target = \
                    discrete_partial_jdcoot(
                        S, T, S_test, T_test,
                        l_source_train, l_source_test,
                        l_target_train, l_target_test,
                        algo=algo, reg=reg, batch_size=batch_size,
                        alphaS=alpha, alphaT=alpha
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "jdcoot", "learning": "partial",
                    "prop_source": prop_values_s, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

                # PARTIAL REFERENCE
                pure_source, pure_target, test_source, test_target = \
                    discrete_partial_reference(
                        S, T, S_test, T_test,
                        l_source_train, l_source_test,
                        l_target_train, l_target_test
                    )
                results.append({
                    "source": source_name, "target": target_name, "fold": fold_idx,
                    "recoding": "reference", "learning": "partial",
                    "prop_source": prop_values_s, "prop_target": prop_values_s,
                    "pure_source": pure_source, "test_source": test_source,
                    "pure_target": pure_target, "test_target": test_target,
                })

# =========================================================
# RESULTS
# =========================================================

df_results = pd.DataFrame(results)

df_summary = (
    df_results
    .groupby(
        ["source", "target", "recoding", "learning", "prop_source", "prop_target"],
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

df_summary.to_excel("results_all_domains_kfold.xlsx", index=False)

print(df_summary)