import numpy as np
import ot
import sklearn

from .coot import cot_numpy
from .coot import random_gamma_init, init_matrix_np
from .losses import loss_crossentropy2, loss_hinge
from .svn_classifier import SVMClassifier


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


"""
models should be compiled with l2 loss
semisupervison :
YA is (nA,nclass) : one-hot-encoded labels of XA : 1 when the label is known, 0 otherwise
YB is (nB,nclass) : one-hot-encoded labels of XB: 1 when the label is known, 0 otherwise
yAtruth,yBtruth : are the true labels of XA and XB (only used to compute the accuracy, never for learning)
shapeA, shapeB : 2D data shapes (if data are images)
reshape_data : if models are CNN (default=False) 
"""


def jdcot_multitask_classif(modelA, modelB, XA, YA, XB, YB, yAtruth, yBtruth, XAtest=[], yAtest=[], XBtest=[],
                            yBtest=[],
                            wA=None, wB=None, vA=None, vB=None, random_init=False, algo='emd', reg=0, algo2='emd',
                            reg2=0, alpha=1,
                            numIterBCD=10, nb_epoch=10, batch_size=10, reshape_data=False, shapeA=None, shapeB=None):
    idxA = np.where(YA == 1)[0]
    idxB = np.where(YB == 1)[0]

    if len(idxB) == 0:
        type_supervision = "Unsupervised"
    else:
        if len(idxA) == XA.shape[0]:
            type_supervision = "Semi"
        else:
            type_supervision = "Partial"

    if type_supervision == "Partial":
        # Initializations
        print("Processing Partial JDCOOT method")
        nA, dA = XA.shape
        nB, dB = XB.shape

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

        if reshape_data:
            XA = XA.reshape(XA.shape[0], shapeA[0], shapeA[1], 1)
            XB = XB.reshape(XB.shape[0], shapeB[0], shapeB[1], 1)
            if len(XAtest):
                XAtest = XAtest.reshape(XAtest.shape[0], shapeA[0], shapeA[1], 1)
            if len(XBtest):
                XBtest = XBtest.reshape(XBtest.shape[0], shapeB[0], shapeB[1], 1)

        # lists of accuracy scores
        trainA = []
        trainB = []
        testA = []
        testB = []

        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs = random_gamma_init(wA, wB)
            Gv = random_gamma_init(vA, vB)

        # models initialization
        # print('initializing models')
        idxA = np.where(YA == 1)[0]  # indices of examples of XA with known labels
        modelA.fit(XA[idxA], YA[idxA], batch_size=10, epochs=nb_epoch,
                   verbose=0)  # we train the classifier with labelled examples only
        YApred = modelA.predict(XA)  # first estimate of XA labels
        perf = 100 * np.mean(np.argmax(YApred, 1) == yAtruth)
        trainA.append(perf)
        # print('pred YA init ',perf)
        YApred[idxA] = YA[idxA]  # injection of known labels in the classifier predictions
        if len(XAtest):
            YAtest = modelA.predict(XAtest)
            yAestim = np.argmax(YAtest, 1)
            perfA = 100 * np.mean(yAestim == yAtest)
            testA.append(perfA)

        idxB = np.where(YB == 1)[0]  # indices of examples of XB with known labels
        modelB.fit(XB[idxB], YB[idxB], batch_size=10, epochs=nb_epoch,
                   verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)  # first estimate of XB labels
        perf = 100 * np.mean(np.argmax(YBpred, 1) == yBtruth)
        trainB.append(perf)
        # print('pred YB init ',perf)
        YBpred[idxB] = YB[idxB]  # injection of known labels in the classifier predictions
        if len(XBtest):
            YBtest = modelB.predict(XBtest)
            yBestim = np.argmax(YBtest, 1)
            perfB = 100 * np.mean(yBestim == yBtest)
            testB.append(perfB)

        results = {}

        # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost = loss_crossentropy2(YApred, YBpred)
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

            # recording source accuracy scores
            yAestim = np.argmax(YApred, 1)
            perfA = 100 * np.mean(yAestim == yAtruth)
            trainA.append(perfA)
            if len(XAtest):
                YAtest = modelA.predict(XAtest)
                yAestim = np.argmax(YAtest, 1)
                perfA = 100 * np.mean(yAestim == yAtest)
                testA.append(perfA)

            YBhat = nB * Gs.T.dot(YApred)
            YBhat[idxB] = YB[idxB]
            modelB.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
            YBpred = modelB.predict(XB)

            #########
            YBpred[idxB] = YB[idxB]

            # recording target accuracy scores
            yBestim = np.argmax(YBpred, 1)
            perfB = 100 * np.mean(yBestim == yBtruth)
            trainB.append(perfB)
            if len(XBtest):
                YBtest = modelB.predict(XBtest)
                yBestim = np.argmax(YBtest, 1)
                perfB = 100 * np.mean(yBestim == yBtest)
                testB.append(perfB)

            # print('YA pred acc :',perfA)
            # print('YB pred acc :',perfB)

            # pl.figure()
            # pl.imshow(fcost)
            # pl.show()

            # function cost update
            # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost = loss_crossentropy2(YApred, YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
            results['YA test acc'] = testA
        if len(XBtest):
            results['YB test acc'] = testB
        return modelA, modelB, results

    if type_supervision == "Semi":
        print("Processing Semi Supervsed JDCOOT method")
        # Initializations
        nA, dA = XA.shape
        nB, dB = XB.shape

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

        if reshape_data:
            XA = XA.reshape(XA.shape[0], shapeA[0], shapeA[1], 1)
            XB = XB.reshape(XB.shape[0], shapeB[0], shapeB[1], 1)
            if len(XAtest):
                XAtest = XAtest.reshape(XAtest.shape[0], shapeA[0], shapeA[1], 1)
            if len(XBtest):
                XBtest = XBtest.reshape(XBtest.shape[0], shapeB[0], shapeB[1], 1)

        # lists of accuracy scores
        trainA = []
        trainB = []
        testA = []
        testB = []

        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs = random_gamma_init(wA, wB)
            Gv = random_gamma_init(vA, vB)

        YApred = YA  # injection of known labels in the classifier predictions

        idxB = np.where(YB == 1)[0]  # indices of examples of XB with known labels
        modelB.fit(XB[idxB], YB[idxB], batch_size=10, epochs=nb_epoch,
                   verbose=0)  # we train the classifier with labelled examples only
        YBpred = modelB.predict(XB)  # first estimate of XB labels

        results = {}

        # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost = loss_crossentropy2(YApred, YBpred)
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

            YApred = YA

            YBhat = nB * Gs.T.dot(YApred)
            YBhat[idxB] = YB[idxB]
            modelB.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
            YBpred = modelB.predict(XB)

            #########
            YBpred[idxB] = YB[idxB]

            # recording target accuracy scores
            yBestim = np.argmax(YBpred, 1)
            perfB = 100 * np.mean(yBestim == yBtruth)
            trainB.append(perfB)
            if len(XBtest):
                YBtest = modelB.predict(XBtest)
                yBestim = np.argmax(YBtest, 1)
                perfB = 100 * np.mean(yBestim == yBtest)
                testB.append(perfB)

            # print('YA pred acc :',perfA)
            # print('YB pred acc :',perfB)

            # pl.figure()
            # pl.imshow(fcost)
            # pl.show()

            # function cost update
            # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost = loss_crossentropy2(YApred, YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
            results['YA test acc'] = testA
        if len(XBtest):
            results['YB test acc'] = testB
        return modelA, modelB, results

    if type_supervision == "Unsupervised":
        print("Processing Unsupervised JDCOOT method")

        # Initializations
        nA, dA = XA.shape
        nB, dB = XB.shape

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

        if reshape_data:
            XA = XA.reshape(XA.shape[0], shapeA[0], shapeA[1], 1)
            XB = XB.reshape(XB.shape[0], shapeB[0], shapeB[1], 1)
            if len(XAtest):
                XAtest = XAtest.reshape(XAtest.shape[0], shapeA[0], shapeA[1], 1)
            if len(XBtest):
                XBtest = XBtest.reshape(XBtest.shape[0], shapeB[0], shapeB[1], 1)

        # lists of accuracy scores
        trainA = []
        trainB = []
        testA = []
        testB = []

        if not random_init:
            Gs = np.ones((nA, nB)) / (nA * nB)  # is (n,n')
            Gv = np.ones((dA, dB)) / (dA * dB)  # is (d,d')
        else:
            Gs = random_gamma_init(wA, wB)
            Gv = random_gamma_init(vA, vB)

        YApred = YA  # injection of known labels in the classifier predictions

        # step 1 : samples coupling optimization
        Ms = alpha * (C_s - np.dot(h1_s, Gv).dot(h2_s.T))  # is (nA,nB)
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

        YApred = YA

        YBhat = nB * Gs.T.dot(YApred)
        modelB.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
        YBpred = modelB.predict(XB)

        results = {}

        # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean')   # is (nA,nB)
        fcost = loss_crossentropy2(YApred, YBpred)
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

            YApred = YA

            YBhat = nB * Gs.T.dot(YApred)
            modelB.fit(XB, YBhat, batch_size=batch_size, epochs=nb_epoch, verbose=0)
            YBpred = modelB.predict(XB)

            # print('YA pred acc :',perfA)
            # print('YB pred acc :',perfB)

            # pl.figure()
            # pl.imshow(fcost)
            # pl.show()

            # function cost update
            # fcost = ot.dist(YApred,YBpred,metric='sqeuclidean') # YA is (nA,nclass), ypred is (nB,nclass), fcost is (nA,nB)
            fcost = loss_crossentropy2(YApred, YBpred)

        results['YA train acc'] = trainA
        results['YB train acc'] = trainB
        if len(XAtest):
            results['YA test acc'] = testA
        if len(XBtest):
            results['YB test acc'] = testB
        return modelA, modelB, results


"""
jdcot multi-task for multi regression problems
npreds : number of parameters to predict (it has to be the same number for both datasets)
yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
YA is (nA,npreds), YB is (nB,npreds) : line of NaN if non observed labels and true values if observed labels (semi supervision)
"""


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
