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

# # Test on CaffeNet4096 and GoogleNet1024 Data

# +
import sklearn
import scipy 
import numpy as np
import pandas as pd
import os
import sys
from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot
from scipy.io import loadmat
featuresToUse = ["CaffeNet4096", "GoogleNet1024"] 
sourceDomainName = ['amazon'] #['caltech10','amazon','webcam']
targetDomainName = ['amazon'] #['caltech10','amazon','webcam']

tests = []
data_source = {}
data_target = {}

min_max_scaler = sklearn.preprocessing.MinMaxScaler()
# Collab
possible_data = loadmat(os.path.join("data/", "features", featuresToUse[0],
                                                 "caltech10" + '.mat'))
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
S_data = [feat, labels]
S_nClass = len(np.unique(labels)) # nb de class in source data
possible_data = loadmat(os.path.join("data/", "features", featuresToUse[1],
                                                 "amazon" + '.mat'))
feat = possible_data['fts'].astype(float)
labels = possible_data['labels'].ravel()
T_data = [feat, labels]
T_nClass = len(np.unique(labels)) # nb de class in target data

source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1
# -


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

    a = np.random.choice(np.arange(len(source)), math.ceil(0.7 * len(source)), replace=False)
    b = np.random.choice(np.arange(len(target)), math.ceil(0.7 * len(target)), replace=False)

    S_test = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
    T_test = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
    S = source.iloc[a, :].reset_index(drop=True)
    T = target.iloc[b, :].reset_index(drop=True)

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
        discrete_unsupervised_jdcoot(S, T, S_test, T_test)

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

source

# +
##j'enregistre les données pour utiliser la fonction get_data
XtotS=S_data[0]
YtotS=S_data[1].reshape(-1,1)
dS=XtotS.shape[1]

XtotT=T_data[0]
YtotT=T_data[1].reshape(-1,1)
dS=XtotT.shape[1]
# -

YtotS

target

# +
import math

a = np.random.choice(np.arange(len(source)), math.ceil(0.7 * len(source)), replace=False)
b = np.random.choice(np.arange(len(target)), math.ceil(0.7 * len(target)), replace=False)

S = source.iloc[np.setdiff1d(np.arange(len(source)), a), :].reset_index(drop=True)
T = target.iloc[np.setdiff1d(np.arange(len(target)), b), :].reset_index(drop=True)
S.loc[:,'Z'] = S.loc[:, 'Z'] - 1
T.loc[:,'Z'] = T.loc[:, 'Z'] - 1
S
# -

S_test = source.iloc[a, :].reset_index(drop=True)
T_test = target.iloc[b, :].reset_index(drop=True)

# ## Performance function

from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
discrete_unsupervised_jdcoot( S, T, S_test, T_test)

from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
discrete_semisupervised_jdcoot( S, T, S_test, T_test, prop_source=0.2)

from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
discrete_partial_jdcoot( S, T, S_test, T_test, prop_source=0.2, prop_target=0.2)

from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
discrete_unsupervised_coot( S, T, S_test, T_test)


# +
# get test set : save examples (nbtest per class) 
# take the last ones so that it's always the same set and we can compare results
# get train set with random
import numpy as np
from random import shuffle
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.model_selection import train_test_split

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


def get_data_2(x,y,nbtrain,nbtest,nseed):
    
    xtrain=np.zeros((0,x.shape[1]))
    ytrain=np.zeros((0))
    xtest=np.zeros((0,x.shape[1]))
    ytest=np.zeros((0))
    print(np.unique(y))
    for i in np.unique(y):
        xi=x[y.ravel()==i,:] # tous les exemples de la classe i

        if len(np.argwhere(y==i)) >= nbtrain+nbtest :
            ni = xi.shape[0] - nbtest    # nb d'exemples de la classe i pouvant servir lors de l'apprentissage
            np.random.seed(nseed)
            idx=np.random.permutation(ni)
          #print(idx[:nbtrain])
        
            xtrain=np.concatenate((xtrain,xi[idx[:nbtrain],:]),0)
            ytrain=np.concatenate((ytrain,i*np.ones(nbtrain)))

            xtest=np.concatenate((xtest,xi[ni:]),0)
            ytest=np.concatenate((ytest,i*np.ones(nbtest)))
 
        else:
            np.random.seed(nseed)
            idx=np.random.permutation(xi.shape[0])
        
            xtrain=np.concatenate((xtrain,xi[idx[:nbtrain]]),0)
            ytrain=np.concatenate((ytrain,i*np.ones(nbtrain)))
 
            xtest=np.concatenate((xtest,xi[idx[nbtrain:]]),0)
            ytest=np.concatenate((ytest,i*np.ones(xi.shape[0]-nbtrain)))
        
    return xtrain,ytrain,xtest,ytest


# semi-supervison
# get nsamples labelled examples for each class and one-hot-encode those labels
# nclass = number of classes in the dataset
# nsamples : number of labelles samples per class
# noLabClass : classes chosen to remain unlabelled
def get_labels(y,nsamples,nclass=10,noLabClass=[]):
    Y = np.zeros((len(y),nclass))
    for c in np.unique(y):
        if c not in noLabClass :
            idx = np.where(y==c)[0]
            Y[idx[:nsamples],int(c)]=1
    return Y

def generateSubset(X, Y, nPerClass):
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        shuffle(idxClass)
        idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
    return (X[idx, :], Y[idx])


# -

from jdcoot.models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from jdcoot.models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from jdcoot.models.discrete_partial_jdcoot import discrete_partial_jdcoot
from jdcoot.models.discrete_unsupervised_coot import discrete_unsupervised_coot
from jdcoot.models.discrete_semisupervised_coot import discrete_semisupervised_coot
from jdcoot.models.discrete_partial_coot import discrete_partial_coot
from jdcoot.models.discrete_semisupervised_reference import discrete_semisupervised_reference
from jdcoot.models.discrete_partial_reference import discrete_partial_reference
from jdcoot.models.continuous_unsupervised_jdcoot import continuous_unsupervised_jdcoot
from jdcoot.models.continuous_unsupervised_coot import continuous_unsupervised_coot

# ## alpha

# +
from jdcoot.utils import xcolumns, discrete_classifier, discrete_accuracy
from jdcoot.coot import init_matrix_np
from jdcoot.losses import loss_crossentropy2
from tf_keras.utils import to_categorical


def one_hot(y, nClass):
    return to_categorical(y, num_classes=nClass)


def one_cold(z_encoded):
    return np.argmax(z_encoded, axis=1)

source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1

classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
nClass = len(classes)

x_source = source.loc[:, xcolumns(source)].values
z_source = source.Z.values

x_target = target.loc[:, xcolumns(target)].values
z_target = target.Z.values

z_target_u= np.unique(z_target)


max_diff2 = max(
    (x_source.max() - x_target.min())**2,
    (x_source.min() - x_target.max())**2
)
max_diff2
# Calcul de toutes les distances euclidiennes

# Distance maximale
#max_distance = np.max(dist_matrix^2)
#print("Max distance:", max_distance)

z1 = np.arange(10)
z2 = np.arange(10)
n_class = 10
Z1 = one_hot(z1, n_class)  # (10, 10)
Z2 = one_hot(z2, n_class)
fcost = np.max(loss_crossentropy2(Z1, Z2))

alpha = fcost/max_diff2
alpha
# -

discrete_unsupervised_jdcoot( S, T, S_test, T_test,alpha=a)

# ## Data labellisation impact

# +
import numpy as np
nbtrain=80
nbtest=20
XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1
XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1

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
alpha_values = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
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
# -

results = []
numRepetitions = 10
nbtrain=80
nbtest=20
a= 0.4
n_source = len(source.Z)
n_target = len(target.Z)
prop_target_values_s = [0.01,0.05,0.1,0.2,0.4]
prop_target_values_p = [0.01,0.05,0.1,0.2,0.4]
XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1
XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1
XtotS.shape
repe=1

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


# +
T['Z'].value_counts()
classes = np.union1d(np.unique(source.Z), np.unique(target.Z))
nClass = len(classes)
nClass
from jdcoot.utils import xcolumns, discrete_classifier, discrete_accuracy

x_source = source.loc[:, xcolumns(source)].values
z_source = source.Z.values

x_target = target.loc[:, xcolumns(target)].values
z_target = target.Z.values

clf = discrete_classifier(target, "relu", "softmax", nClass)

# -

discrete_unsupervised_jdcoot(S,T, S_test, T_test,alpha=a)

discrete_unsupervised_coot( S, T, S_test, T_test,alpha=a)

target.shape

# +
import math

results = []
numRepetitions = 1
nbtrain=800
nbtest=200
a= 0.4
repe=1
n_source = len(source.Z)
n_target = len(target.Z)
prop_target_values_s = [0.01,0.05,0.1,0.2,0.4]
prop_target_values_p = [0.01,0.05,0.1,0.2,0.4]
XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1
XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1


#for repe in range(numRepetitions):
#    print('num repe :', repe + 1)
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

    # =========================================================
    # UNSUPERVISED
    # =========================================================
#for recoding, func in [
#        ("jdcoot", discrete_unsupervised_jdcoot),
 #       ("coot", discrete_unsupervised_coot),
 #   ]:
pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S, T,alpha=a)

test_target


# +
pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S, T,alpha=a)

test_target
# -

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

# +
pure_source, pure_target, test_source, test_target = discrete_unsupervised_coot(S, T, S, T)


# -

test_target

df_results = pd.DataFrame(results)
df_results
df_results.to_excel("results_mean.xlsx", index=False)

# +
source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1

a=np.random.choice(np.arange(len(source.index)),math.ceil(0.7*len(source.index)),replace=False)
b=np.random.choice(np.arange(len(target.index)),math.ceil(0.7*len(target.index)),replace=False)
S_test=source.iloc[a,:].reset_index(drop=True)
T_test=target.iloc[b,:].reset_index(drop=True)

S=source.iloc[np.setdiff1d(np.arange(len(source.index)),a),:].reset_index(drop=True)
T=target.iloc[np.setdiff1d(np.arange(len(target.index)),b),:].reset_index(drop=True)
T_test.shape
# -

pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T,  S_test, T_test, alpha=0.66)


# +
import math
results = []
numRepetitions = 10
a = 0.5

prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]



source=pd.DataFrame(np.concatenate((S_data[0],S_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(S_data[0].shape[1])]+['Z'])
target=pd.DataFrame(np.concatenate((T_data[0],T_data[1].reshape(-1,1)),axis=1),columns=['X'+str(i) for i in range(T_data[0].shape[1])]+['Z'])

source.loc[:,'Z'] = source.loc[:, 'Z'] - 1
target.loc[:,'Z'] = target.loc[:, 'Z'] - 1

for repe in range(numRepetitions):
    print("num repe :", repe + 1)

    a=np.random.choice(np.arange(len(source.index)),math.ceil(0.7*len(source.index)),replace=False)
    b=np.random.choice(np.arange(len(target.index)),math.ceil(0.7*len(target.index)),replace=False)
    S=source.iloc[a,:].reset_index(drop=True)
    T=target.iloc[b,:].reset_index(drop=True)

    S_test=source.iloc[np.setdiff1d(np.arange(len(source.index)),a),:].reset_index(drop=True)
    T_test=target.iloc[np.setdiff1d(np.arange(len(target.index)),b),:].reset_index(drop=True)
    # =========================================================
    # UNSUPERVISED
    # =========================================================

    # COOT (sans alpha)
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_coot(S, T, S_test, T_test)

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
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T,  S_test, T_test, alpha = 0.4)

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
# -

 # =========================================================
    # SEMI-SUPERVISED
    # =========================================================
    for prop_target in prop_target_values_s:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_jdcoot(
                S, T, S_test, T_test,
                alpha=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = \
            discrete_semisupervised_coot(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_reference(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

    # =========================================================
    # PARTIAL
    # =========================================================
    for prop_target in prop_target_values_p:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(
                S, T, S_test, T_test,
                alpha=0.5,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })    
 

df_summary

# +
results = []
numRepetitions = 10
nbtrain = 80
nbtest = 20
a = 0.01

prop_target_values_s = [0.01, 0.05, 0.1, 0.2, 0.4]
prop_target_values_p = [0.01, 0.05, 0.1, 0.2, 0.4]

XtotS = np.array(S_data[0], dtype=float)
YtotS = np.array(S_data[1], dtype=int).reshape(-1, 1) - 1

XtotT = np.array(T_data[0], dtype=float)
YtotT = np.array(T_data[1], dtype=int).reshape(-1, 1) - 1

for repe in range(numRepetitions):
    print("num repe :", repe + 1)

    XS1, yS1, XStest, yStest = get_data_2(XtotS, YtotS, nbtrain, nbtest, repe)
    XT1, yT1, XTtest, yTtest = get_data_2(XtotT, YtotT, nbtrain, nbtest, repe)

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

    # JDCOOT
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_jdcoot(S, T, S, T, alpha=a)

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

    # COOT (sans alpha)
    pure_source, pure_target, test_source, test_target = discrete_unsupervised_coot(S, T, S, T)

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

   # =========================================================
    # SEMI-SUPERVISED
    # =========================================================
    for prop_target in prop_target_values_s:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_jdcoot(
                S, T, S_test, T_test,
                alpha=a,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = \
            discrete_semisupervised_coot(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_semisupervised_reference(
                S, T, S_test, T_test,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "semisupervised",
            "prop_source": 1,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

    # =========================================================
    # PARTIAL
    # =========================================================
    for prop_target in prop_target_values_p:

        # JDCOOT
        pure_source, pure_target, test_source, test_target = discrete_partial_jdcoot(
                S, T, S_test, T_test,
                alpha=a,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "jdcoot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # COOT
        pure_source, pure_target, test_source, test_target = discrete_partial_coot(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "coot",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })

        # Reference
        pure_source, pure_target, test_source, test_target = discrete_partial_reference(
                S, T, S_test, T_test,
                prop_source=0.5,
                prop_target=prop_target
            )

        results.append({
            "repetition": repe,
            "recoding": "reference",
            "learning": "partial",
            "prop_source": 0.5,
            "prop_target": prop_target,
            "pure_source": pure_source,
            "test_source": test_source,
            "pure_target": pure_target,
            "test_target": test_target,
        })


df_results = pd.DataFrame(results)
df_results

print(df_results)
df_summary.to_excel("results_mean.xlsx", index=False)

# +
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
# -

df_summary

df_summary.to_excel("results_mean.xlsx", index=False)

# +
alpha_grid = np.linspace(0, 1, 11)  # ex: [0.0, 0.1, ..., 1.0]


best_alpha = None
best_score = -np.inf  # ou +np.inf si MSE

numRepetitions_alpha = 5  

for a in alpha_grid:
    scores = []

    for repe in range(numRepetitions_alpha):
        XS1, yS1, XStest, yStest = get_data_2(XtotS, YtotS, nbtrain, nbtest, repe)
        XT1, yT1, XTtest, yTtest = get_data_2(XtotT, YtotT, nbtrain, nbtest, repe)

        S = pd.DataFrame(np.c_[XS1, yS1],
                         columns=[f'X{i}' for i in range(XS1.shape[1])] + ['Z'])
        T = pd.DataFrame(np.c_[XT1, yT1],
                         columns=[f'X{i}' for i in range(XT1.shape[1])] + ['Z'])
        S_test = pd.DataFrame(np.c_[XStest, yStest],
                              columns=[f'X{i}' for i in range(XStest.shape[1])] + ['Z'])
        T_test = pd.DataFrame(np.c_[XTtest, yTtest],
                              columns=[f'X{i}' for i in range(XTtest.shape[1])] + ['Z'])

        _, _, _, test_target = discrete_unsupervised_jdcoot(
            S, T, S_test, T_test, alpha=a
        )

        scores.append(test_target)

    mean_score = np.mean(scores)

    if mean_score > best_score:
        best_score = mean_score
        best_alpha = a

print("Best alpha:", best_alpha)
print("Best score:", best_score)


# +
import math
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
# -

df_summary
