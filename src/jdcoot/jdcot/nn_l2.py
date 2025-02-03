import numpy as np
import ot
import sklearn

from ..coot import cot_numpy
from ..coot import random_gamma_init, init_matrix_np
from ..losses import loss_crossentropy2, loss_hinge
from ..svn_classifier import SVMClassifier



# model is a nn compiled with l2 loss
# YA is (nA,nclass) --> one-hot-encoded labels of XA
# yB is (nB,) --> labels of XB
# XBtest,yBtest : test target data (only to evaluate the model with unseen data over iterations)

def jdcot_nn_l2(model, XA, YA, XB, yB=[], XBtest=[], yBtest=[],
                wA=None, wB=None, vA=None, vB=None, random_init=False,
                algo='emd', reg=0, algo2='emd', reg2=0, alpha=1,
                numIterBCD=10, nb_epoch=10, batch_size=10):
    """

    Args:
      model:
      XA:
      YA:
      XB:
      yB:
      XBtest:
      yBtest:
      wA:
      wB:
      vA:
      vB:
      random_init:
      algo:
      reg:
      algo2:
      reg2:
      alpha:
      numIterBCD:
      nb_epoch:
      batch_size:

    Returns:

    """

    # Initializations
    nA, dA = XA.shape
    nB, dB = XB.shape

    LPS = []  # label propagation score
    train_score = []
    test_score = []

    if vA is None:
        vA = np.ones(dA) / dA  # is (d,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (d',)
    if wA is None:
        wA = np.ones(nA) / nA  # is (n,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (n',)

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

    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    # function cost initialization
    fcost = np.zeros((nA, nB))  # is (nA,nB)

    # BCD iterations : laternate samples and variables mappings optimisations
    for k in range(numIterBCD):
        # print('num iter BCD:',k+1)

        # step 1 : samples coupling optimisation
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + fcost  # is (nA,nB)
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : variables coupling optimisation
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        # label propagation : estimated labels of XB using transport map on samples from A
        YBhat = nB * Gs.T.dot(YA)  # soft labels - is (nB,nclass)
        yestim = np.argmax(YBhat, 1)  # classification decision --> hard labels - is (nB,)

        # training model on target data with estimated labels
        model.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        ypred = model.predict(XB)
        if len(XBtest):
            yval = model.predict(XBtest)
            yval = np.argmax(yval, 1)

        # recording scores
        if len(yB):
            LPS.append(list(yestim - yB).count(0) / nB * 100)
            train_score.append(list(np.argmax(ypred, 1) - yB).count(0) / nB * 100)
            # print('label propagation score : ',LPS[-1])
            # print('train score : ',train_score[-1])
        if len(yBtest):
            test_score.append(list(yval - yBtest).count(0) / len(yBtest))
            # print('test score :',list(yval-YBtest).count(0)/XBtest.shape[0])

        # pl.figure()
        # pl.imshow(fcost)
        # pl.show()

        if k > 1:
            sav_fcost.append(np.sum(Gs * fcost))
            sav_totalcost.append(np.sum(Gs * Ms))

        # function cost update
        fcost = ot.dist(YA, ypred, metric='sqeuclidean')  # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)

    if len(yB):
        results['ypred0'] = ypred
        results['ypred'] = np.argmax(ypred, 1)
        results['labProp'] = LPS
        results['train'] = train_score
    if len(yBtest):
        results['test'] = test_score
    results['clf'] = model
    results['fcost'] = sav_fcost
    results['totalcost'] = sav_totalcost

    return model, results

