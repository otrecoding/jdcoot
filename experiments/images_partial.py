import sklearn
import scipy 
import numpy as np
import pandas as pd
import os
import sys
import math


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
from scipy.io import loadmat
from jdcoot.models.discrete_partial_jdcoot3 import discrete_partial_jdcoot3
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_partial_jdcoot2 import discrete_partial_jdcoot2
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_partial_coot2   import discrete_partial_coot2
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference


featuresToUse = ["CaffeNet4096", "GoogleNet1024"] 
#featuresToUse = ["CaffeNet4096", "CaffeNet4096"] 
import os
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
import sklearn.preprocessing

# =========================================================
# PARAMETERS
# =========================================================

sourceDomainName = ['amazon', 'caltech10', 'webcam']
targetDomainName = ['caltech10', 'amazon', 'webcam']

numRepetitions = 10
algo = "sinkhorn"
reg = 1
batch_size = 20
alpha = 1.5
prop_values = [1, 2, 3, 4, 5]

results = []

# =========================================================
# LOOP OVER ALL SOURCE / TARGET COMBINATIONS
# =========================================================

for source_name in sourceDomainName:

    for target_name in targetDomainName:

        # Skip identical domains if needed
        if source_name == target_name:
            continue

        print("=" * 60)
        print(f"SOURCE = {source_name} | TARGET = {target_name}")
        print("=" * 60)

        # =========================================================
        # LOAD SOURCE
        # =========================================================

        possible_data = loadmat(
            os.path.join(
                "data/",
                "features",
                featuresToUse[0],
                source_name + '.mat'
            )
        )

        feat = possible_data['fts'].astype(float)
        labels = possible_data['labels'].ravel()

        S_data = [feat, labels]

        # =========================================================
        # LOAD TARGET
        # =========================================================

        possible_data = loadmat(
            os.path.join(
                "data/",
                "features",
                featuresToUse[1],
                target_name + '.mat'
            )
        )

        feat = possible_data['fts'].astype(float)
        labels = possible_data['labels'].ravel()

        T_data = [feat, labels]

        # =========================================================
        # DATAFRAMES
        # =========================================================

        source = pd.DataFrame(
            np.concatenate(
                (S_data[0], S_data[1].reshape(-1, 1)),
                axis=1
            ),
            columns=['X' + str(i) for i in range(S_data[0].shape[1])] + ['Z']
        )

        target = pd.DataFrame(
            np.concatenate(
                (T_data[0], T_data[1].reshape(-1, 1)),
                axis=1
            ),
            columns=['X' + str(i) for i in range(T_data[0].shape[1])] + ['Z']
        )

        source.loc[:, 'Z'] = source.loc[:, 'Z'] - 1
        target.loc[:, 'Z'] = target.loc[:, 'Z'] - 1

        # =========================================================
        # REPETITIONS
        # =========================================================

        for repe in range(numRepetitions):

            print(f"Repetition {repe + 1}")

            a = np.random.choice(
                np.arange(len(source)),
                math.ceil(0.8 * len(source)),
                replace=False
            )

            b = np.random.choice(
                np.arange(len(target)),
                math.ceil(0.8 * len(target)),
                replace=False
            )

            S_test = source.iloc[
                np.setdiff1d(np.arange(len(source)), a), :
            ].reset_index(drop=True)

            T_test = target.iloc[
                np.setdiff1d(np.arange(len(target)), b), :
            ].reset_index(drop=True)

            S = source.iloc[a, :].reset_index(drop=True)
            T = target.iloc[b, :].reset_index(drop=True)

            # =====================================================
            # UNSUPERVISED
            # =====================================================

            # COOT


            # =====================================================
            # SEMI-SUPERVISED + PARTIAL
            # =====================================================

            for prop_values_s in prop_values:

                # TARGET LABELS
                l_target_train = []

                for z in np.unique(T['Z']):

                    idx = np.where(T['Z'] == z)[0]

                    chosen = np.random.choice(
                        idx,
                        size=prop_values_s,
                        replace=False
                    )

                    l_target_train.extend(chosen)

                l_target_train = np.array(l_target_train)

                l_target_test = np.setdiff1d(
                    np.arange(len(T['Z'])),
                    l_target_train
                )

                # -------------------------------------------------
                # SEMISUPERVISED COOT
                # -------------------------------------------------

         
                # -------------------------------------------------
                # SOURCE LABELS
                # -------------------------------------------------

                l_source_train = []

                for z in np.unique(S['Z']):

                    idx = np.where(S['Z'] == z)[0]

                    chosen = np.random.choice(
                        idx,
                        size=prop_values_s,
                        replace=False
                    )

                    l_source_train.extend(chosen)

                l_source_train = np.array(l_source_train)

                l_source_test = np.setdiff1d(
                    np.arange(len(S['Z'])),
                    l_source_train
                )

                # -------------------------------------------------
                # PARTIAL COOT
                # -------------------------------------------------

 
                pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(
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
                batch_size=batch_size,
                alpha=alpha
            )

                results.append({
                    "source": source_name,
                    "target": target_name,
                    "repetition": repe,
                    "recoding": "jdcoot",
                    "learning": "partial",
                    "prop_source": 1,
                    "prop_target": prop_values_s,
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
        [
            "source",
            "target",
            "recoding",
            "learning",
            "prop_source",
            "prop_target"
        ],
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

df_summary.to_excel("results_all_domains.xlsx", index=False)

print(df_summary)