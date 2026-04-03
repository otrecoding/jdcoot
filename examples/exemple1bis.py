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
import sklearn
import scipy 
import numpy as np
import pandas as pd
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

sourceDomainName = ['amazon'] #['caltech10','amazon','webcam']
targetDomainName = ['amazon'] #['caltech10','amazon','webcam']

tests = []
data_source = {}
data_target = {}

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# Collab
possible_data = scipy.io.loadmat('caltech10_caffe.mat')
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
S_data = [feat, labels]
S_nClass = len(np.unique(labels)) # nb de class in source data
possible_data = scipy.io.loadmat('caltech10_google.mat')
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
T_data = [feat, labels]
T_nClass = len(np.unique(labels)) # nb de class in target data

source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1


# -

def get_data(x, y, nbtrain, nbtest, nseed):
 
    y = y.ravel()
    n = x.shape[0]

    if nbtrain + nbtest > n:
        raise ValueError("nbtrain + nbtest exceeds total number of samples")

    np.random.seed(nseed)
    idx = np.random.permutation(n)

    train_idx = idx[:nbtrain]
    test_idx = idx[nbtrain:nbtrain + nbtest]

    xtrain = x[train_idx]
    ytrain = y[train_idx]

    xtest = x[test_idx]
    ytest = y[test_idx]

    return xtrain, ytrain, xtest, ytest



# +
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot



# +
import numpy as np
nbtrain=80
nbtest=20
XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1
XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1
repe = 1

XS1,yS1,XStest,yStest = get_data(XtotS,YtotS,nbtrain,nbtest,repe)
XT1,yT1,XTtest,yTtest = get_data(XtotT,YtotT,nbtrain,nbtest,repe)

    #XS, yS = generateSubset(XS1, yS1, perClassSource)
    #XT, yT = generateSubset(XT1, yT1, perClassSource)
S = pd.DataFrame(np.c_[XS1, yS1],
                     columns=['X' + str(i) for i in range(XS1.shape[1])] + ['Z'])
T = pd.DataFrame(np.c_[XT1, yT1],
                     columns=['X' + str(i) for i in range(XT1.shape[1])] + ['Z'])
S_test = pd.DataFrame(np.c_[XStest, yStest],
                          columns=['X' + str(i) for i in range(XStest.shape[1])] + ['Z'])
T_test = pd.DataFrame(np.c_[XTtest, yTtest],
                          columns=['X' + str(i) for i in range(XTtest.shape[1])] + ['Z'])


# Exemple de valeurs à tester pour alpha
alpha_values = np.arange(0, 4.01, 0.5)   # 0, 0.1, 0.2, ..., 1.0
best_alpha = None
best_score = -np.inf  # ou 0 selon ta métrique

for a in alpha_values:

    pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(source, target, source, target, alpha=a)

    score = test_target  
    
    if score > best_score:
        best_score = score
        best_alpha = a

print("Meilleur alpha :", best_alpha)
print("Score associé :", best_score)

# +
results = []
numRepetitions = 10
nbtrain = 800
nbtest = 200
a = 0.4

prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]

XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1

XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1

for repe in range(numRepetitions):
    print("num repe :", repe + 1)

    XS1, yS1, XStest, yStest = get_data(XtotS, YtotS, nbtrain, nbtest, repe)
    XT1, yT1, XTtest, yTtest = get_data(XtotT, YtotT, nbtrain, nbtest, repe)

    S = pd.DataFrame(
        np.c_[XS1, yS1],
        columns=[f"X{i}" for i in range(XS1.shape[1])] + ["Z"]
    )

    T = pd.DataFrame(
        np.c_[XT1, yT1],
        columns=[f"X{i}" for i in range(XT1.shape[1])] + ["Z"]
    )

    S_test = pd.DataFrame(
        np.c_[XStest, yStest],
        columns=[f"X{i}" for i in range(XStest.shape[1])] + ["Z"]
    )

    T_test = pd.DataFrame(
        np.c_[XTtest, yTtest],
        columns=[f"X{i}" for i in range(XTtest.shape[1])] + ["Z"]
    )

    # =========================================================
    # UNSUPERVISED
    # =========================================================
     # COOT
    pure_source, pure_target, test_source, test_target = \
        discrete_unsupervised_coot(S, T, S_test, T_test)

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
    pure_source, pure_target, test_source, test_target = \
        discrete_unsupervised_jdcoot(S, T, S_test, T_test, alpha=a)

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

   
import pandas as pd

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

df_summary
  
# -

df_results


