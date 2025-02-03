import numpy as np
import ot
import sklearn

from ..coot import cot_numpy
from ..coot import random_gamma_init, init_matrix_np
from ..losses import loss_crossentropy2, loss_hinge
from ..svn_classifier import SVMClassifier


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


