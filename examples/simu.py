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
from sklearn.model_selection import train_test_split
# -

from jdcoot.models.continuous_unsupervised_jdcoot import continuous_unsupervised_jdcoot
from jdcoot.models.continuous_unsupervised_coot import continuous_unsupervised_coot
from jdcoot.models.continuous_semisupervised_jdcoot import continuous_semisupervised_jdcoot
from jdcoot.models.continuous_semisupervised_coot import continuous_semisupervised_coot
from jdcoot.models.continuous_partial_jdcoot3 import continuous_partial_jdcoot3
from jdcoot.models.continuous_partial_coot2 import continuous_partial_coot2
from jdcoot.scenario import generate_data

results = []
numRepetitions = 10
a = 0.00001
algo = "sinkhorn"
reg = 1
prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]

data = generate_data()
source = data[0]
target = data[1]

pure_source, pure_target, test_source, test_target = continuous_unsupervised_coot(*data, algo=algo, reg=reg, alpha=a)
test_target

pure_source, pure_target, test_source, test_target = continuous_unsupervised_jdcoot(*data, algo=algo, reg=reg, alpha=a)
test_target

y_target = target.Y.values[:, np.newaxis]
n_target = len(y_target)
prop_target = 0.1
l_target_train, l_target_test = train_test_split(
        np.arange(n_target), train_size=prop_target)

pure_source, pure_target, test_source, test_target = continuous_semisupervised_coot(*data,l_target_train, l_target_test, algo=algo, reg=1)
test_target

pure_source, pure_target, test_source, test_target = continuous_semisupervised_jdcoot(*data,l_train=l_target_train, l_test=l_target_test, algo=algo, reg=reg, alpha=a)
test_target 

prop_source=0.95
prop_target=0.005
y_target = target.Y.values[:, np.newaxis]
n_target = len(y_target)
y_source = source.Y.values[:, np.newaxis]
n_source = len(y_source)
l_source_train, l_source_test = train_test_split(
   np.arange(n_source), train_size=prop_source
)
l_target_train, l_target_test = train_test_split(
    np.arange(n_target), train_size=prop_target
)

pure_source, pure_target, test_source, test_target = continuous_partial_coot2(*data,l_source_train, l_source_test,l_target_train, l_target_test ,algo=algo, reg=reg,)
test_target

pure_source, pure_target, test_source, test_target = continuous_partial_jdcoot3(*data, l_source_train, l_source_test,l_target_train, l_target_test,algo=algo, reg=reg, alpha=a,)
test_target



# +
results = []
numRepetitions = 10
a = 0.5

prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]

from jdcoot.scenario import generate_data

source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1

for repe in range(numRepetitions):
    print("num repe :", repe + 1)
    data = generate_data()
    # =========================================================
    # UNSUPERVISED
    # =========================================================

    # COOT (sans alpha)
    pure_source, pure_target, test_source, test_target = continuous_unsupervised_coot(*data)
    print(test_target)
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
    pure_source, pure_target, test_source, test_target = continuous_unsupervised_jdcoot(*data)
    print(test_target)
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
