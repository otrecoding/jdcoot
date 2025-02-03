import numpy as np
import ot
import sklearn

from ..coot import cot_numpy
from ..coot import random_gamma_init, init_matrix_np
from ..losses import loss_crossentropy2, loss_hinge
from ..svn_classifier import SVMClassifier

def jdcot_multitask_reg(modelA, modelB, XA, YA, XB, YB, yAtruth, yBtruth, XAtest=[], XBtest=[], yAtest=[], yBtest=[],
                        shapeA=None, shapeB=None, wA=None, wB=None, vA=None, vB=None, random_init=False,
                        algo='emd', reg=0, algo2='emd', reg2=0, alpha=1,
                        numIterBCD=10, nb_epoch=10, batch_size=10, reshape_data=True):

    trainA = []
    trainB = []
    testA = []
    testB = []
    results = {}

    # Initializations
    nA, dA = XA.shape
    nB, dB = XB.shape

    ######

    shapeA = [dA, 1]  # 1 can be replaced by the number of variables of Y to pred
    shapeB = [dB, 1]
    #####

    if vA is None:
        vA = np.ones(dA) / dA  # is (d,)
    if vB is None:
        vB = np.ones(dB) / dB  # is (d',)
    if wA is None:
        wA = np.ones(nA) / nA  # is (n,)
    if wB is None:
        wB = np.ones(nB) / nB  # is (n',)

    # original losses
    C_s, h1_s, h2_s = init_matrix_np(XA, XB, vA, vB)
    C_v, h1_v, h2_v = init_matrix_np(XA.T, XB.T, wA, wB)

    if not random_init:
        Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
        Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
    else:
        Gs = random_gamma_init(wA, wB)
        Gv = random_gamma_init(vA, vB)

    if reshape_data:
        XA = XA.reshape(XA.shape[0], shapeA[0], shapeA[1], 1)
        XB = XB.reshape(XB.shape[0], shapeB[0], shapeB[1], 1)
        # XAtest = XAtest.reshape(XAtest.shape[0],shapeA[0],shapeA[1],1)
        # XBtest = XBtest.reshape(XBtest.shape[0],shapeB[0],shapeB[1],1)

    # models initialization
    # print('initializing models')
    idxA = np.where(~np.isnan(YA))[0]  # indices of examples of XA with known labels

    modelA.fit(XA[idxA], YA[idxA], batch_size=10, epochs=nb_epoch,
               verbose=0)  # we train the model with labelled examples only
    YApred = modelA.predict(XA)  # first estimate of XA labels
    perf = modelA.evaluate(XA, yAtruth)
    trainA.append(perf)
    # print('pred YA init ',perf)
    # print(YApred.shape)
    # print(YA)
    # print(idxA)
    YApred[idxA] = YA[idxA]  # injection of known labels in the model predictions

    idxB = np.where(~np.isnan(YB))[0]  # indices of examples of XB with known labels

    if len(idxB) == 0:  # first initialisation in case of unsupervised treatment (COOT on YB)

        M_lin = None

        Ts, Tv, cost = cot_numpy(X1=XA,
                                 X2=XB,
                                 niter=100, C_lin=M_lin,
                                 algo='sinkhorn', reg=1,
                                 algo2='emd', verbose=False)

        # Target estimation
        # PLUS DE OH ENC
        YBpred = nB * np.dot(Ts.T, YApred).reshape((-1, 1))
        # print(YBpred)


    else:
        modelB.fit(XB[idxB], YB[idxB], batch_size=10, epochs=nb_epoch,
                   verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)  # first estimate of XB labels
        perf = modelB.evaluate(XB, yBtruth)
        trainB.append(perf)
        # print('pred YB init ',perf)
        YBpred[idxB] = YB[idxB]  # injection of known labels in the model predictions

    # print(YApred)
    # print(YBpred)
    fcost = ot.dist(YApred, YBpred, metric='sqeuclidean')  # is (nA,nB)
    # fcost = np.zeros((nA,nB))

    cost = []

    for k in range(numIterBCD):
        # print('\n num iter BCD:',k+1)

        # step 1 : samples coupling optimization 
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T)) + fcost  # is (nA,nB)
        if algo == 'emd':
            Gs = ot.emd(wA, wB, Ms, numItermax=1e7)
        elif algo == 'sinkhorn':
            Gs = ot.sinkhorn(wA, wB, Ms, reg)

        # step 2 : features coupling optimization     
        Mv = C_v - np.dot(h1_v, Gs).dot(h2_v.T)  # is (dA,dB)
        if algo2 == 'emd':
            Gv = ot.emd(vA, vB, Mv, numItermax=1e7)
        elif algo2 == 'sinkhorn':
            Gv = ot.sinkhorn(vA, vB, Mv, reg2)

        # estimated labels of XA using transport map on samples
        YAhat = nA * Gs.dot(YBpred)
        YAhat[idxA] = YA[idxA]

        # update label estimate with models predictions
        modelA.fit(XA, YAhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        YApred = modelA.predict(XA)
        YApred[idxA] = YA[idxA]
        perfA = modelA.evaluate(XA, yAtruth)
        trainA.append(perfA)

        YBhat = nB * Gs.T.dot(YApred)
        YBhat[idxB] = YB[idxB]
        modelB.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        YBpred = modelB.predict(XB)
        YBpred[idxB] = YB[idxB]
        perfB = modelB.evaluate(XB, yBtruth)
        trainB.append(perfB)

        # print('YA pred acc :',perfA)
        # print('YB pred acc :',perfB)

        # pl.figure()
        # pl.imshow(fcost)
        # pl.show()

        # function cost
        fcost = ot.dist(YApred, YBpred,
                        metric='sqeuclidean')  # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)

    results['YA train acc'] = trainA
    results['YB train acc'] = trainB
    # results['YA test acc'] = testA
    # results['YB test acc'] = testB
    return modelA, modelB, results
