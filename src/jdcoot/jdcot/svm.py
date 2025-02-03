import numpy as np
import ot
import sklearn

from ..coot import cot_numpy
from ..coot import random_gamma_init, init_matrix_np
from ..losses import loss_crossentropy2, loss_hinge
from ..svn_classifier import SVMClassifier


def jdcot_svm(XA, yA, XB, yB=[], wA=None, wB=None, vA=None, vB=None,
              algo='emd', reg=0, algo2='emd', reg2=0,
              gamma_g=1, numIterBCD=10, alpha=1,
              lambd=1e1, ktype='linear', random_init=False):
    """
    XA : source domain data - is (nA,dA)
    yA : source labels
    XB : target domain data is (nB,dB)
    yB is optionnal, target labels just to measure performances of the method along iterations
    wA : weights of source samples (default=None which means uniform distribution if not specified)
    wB : weights of target samples (default=None which means uniform distribution if not specified)
    vA : weights of source features (default=None which means uniform distribution if not specified)
    vB : weights of target features (default=None which means uniform distribution if not specified)
    gamma: RBF kernel param (default = 1)
    numIterBCD: number of Iterations for BCD (default = 10)
    alpha: ponderation between ground cost + function cost
    algo/algo2 : choice of algorithm for transport computation for respectively the samples and features transport (default="emd")
    reg/reg2 : choice of the regularization parameter if algo/algo2 is "sinkhorn" (default = 0)
    random_init : wether the transportation plans are randomly initiated (default=False)
    """

    # Initializations
    nA, dA = XA.shape
    nB, dB = XB.shape

    if vA is None:
        vA = np.ones(dA) / dA  # is (dA,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (dB,)
    if wA is None:
        wA = np.ones(nA) / nA  # is (nA,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (nB,)

    if not random_init:
        Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
        Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
    else:
        Gs = random_gamma_init(wA, wB)
        Gv = random_gamma_init(vA, vB)

    # original loss 
    C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)
    cost = np.inf

    Gsold = Gs
    Gvold = Gv
    costold = cost

    # classifier    
    g = SVMClassifier(lambd)

    # compute kernels
    if ktype == 'rbf':
        Kt = sklearn.metrics.pairwise.rbf_kernel(XB, gamma=gamma_g)
        # Ks=sklearn.metrics.pairwise.rbf_kernel(X,gamma=gamma_g)
    else:
        Kt = sklearn.metrics.pairwise.linear_kernel(XB)
        # Ks=sklearn.metrics.pairwise.linear_kernel(X)

    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}
    ypred = np.zeros(yA.shape)
    Chinge = np.zeros((nA, nB))

    # do it only if the final labels were given
    if len(yB):
        TBR.append(np.mean(yB == np.argmax(ypred, 1)))

    k = 0
    while (k < numIterBCD):

        k += 1
        # step 1 : samples coupling optimization 
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + Chinge  # is (n,n')
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (d,d')
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        if k > 1:
            sav_fcost.append(np.sum(Gs * Chinge))
            sav_totalcost.append(np.sum(Gs * Ms))

        # step 3 : prediction function optimization
        Yst = nB * Gs.T.dot((yA + 1) / 2.)  # label propagation
        if len(yB):
            yestim = np.argmax(Yst, 1)
            ytruth = np.argmax(yB, 1)
            # print(list(yestim-ytruth).count(0)/nB)
        # Yst=ntest*G.T.dot(y_f)
        g.fit(Kt, Yst)
        ypred = g.predict(Kt)

        Chinge = loss_hinge(yA, ypred)
        # Chinge=SVMclassifier.loss_hinge(y_f*2-1,ypred*2-1)

        # C=alpha*C0+Chinge

        if len(yB):
            TBR1 = np.mean(yB == np.argmax(ypred, 1))
            TBR.append(TBR1)

    results['ypred'] = np.argmax(ypred, 1)
    if len(yB):
        results['TBR'] = TBR

    results['clf'] = g
    results['Gs'] = Gs
    results['Gv'] = Gv
    results['fcost'] = sav_fcost
    results['totalcost'] = sav_totalcost

    return g, results


