import math

import numpy as np
import ot
import tf_keras
from sklearn.preprocessing import OneHotEncoder as onehot
from tf_keras.layers import Dense

from .comp import comp_
from .comp import comp_regression
from .jdcot import cot_numpy
from .jdcot import jdcot_multitask_classif
from .jdcot import jdcot_multitask_reg

# -

# # Performance Measurement

# +


def Performance(Data, Data_test=None, Objective_Variable='discrete', algo='both', type_supervision='unsupervised',
                Balance=True, alpha=None, Labelled_Proportion_Target=None, Labelled_Proportion_Source=None, ):
    """
    Performance : Compute efficiency of the methods with the given data. Can be applied to any data if the format is respected.

    Input : Data (tuple) : Learning data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

            Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

                                If None, Learning data is splited into 30% of test data and 70% of learning data

            Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation
                                                                    if 'continuous', measure the performance for regression

            algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT
                                                        if 'JDCOOT', measure the performance for JDCOOT
                                                        if 'both',  measure the performance for COOT and JDCOOT

            type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                           if 'semi-supervised', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and all source observations are labelled
                                                                                                           if 'partial', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and Labelled_Proportion_Source*100 % of the observations of source are labelled
                                                                                                           if 'cross-partial', measure the performance such that Labelled_Proportion_Target*100 % of the observations of target are labelled and Labelled_Proportion_Source*100 % of the observations of source are labelled. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset

            Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data.

            alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())

            Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.

            Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis.


    Output : tuple of 6 elements : Pure (transport) performance of COOT,
                                   Pure (transport) performance of JDCOOT,
                                   Test (trained model applied to test data) performance of COOT,
                                   Test (trained model applied to test data) performance of JDCOOT,
                                   Test (trained model applied to test data) performance of REFERENCE,
                                   Pure (transport) performance of REFERENCE
    """


    S = Data[0]
    T = Data[1]

    if Data_test == None:
        a = np.random.choice(np.arange(len(S.index)), math.ceil(0.7 * len(S.index)), replace=False)
        b = np.random.choice(np.arange(len(T.index)), math.ceil(0.7 * len(T.index)), replace=False)
        S_test = S.iloc[a, :].reset_index(drop=True)
        T_test = T.iloc[b, :].reset_index(drop=True)

        S = S.iloc[np.setdiff1d(np.arange(len(S.index)), a), :].reset_index(drop=True)
        T = T.iloc[np.setdiff1d(np.arange(len(T.index)), b), :].reset_index(drop=True)
    else:
        S_test = Data_test[0]
        S_test = S_test.loc[:, S.columns]
        T_test = Data_test[1]
        T_test = T_test.loc[:, T.columns]

        # print(S)
        # print(S_test)

    if Balance == True and (Objective_Variable == 'discrete' or Objective_Variable == 'both'):
        if len(np.unique(S['Z'])) > 2:

            del_idx = np.array([])
            for k in np.unique(S['Z']):
                if sum(S['Z'] == k) < 0.01 * len(S['Z']):
                    del_idx = np.append(del_idx, k)
            S = S.loc[~np.in1d(S['Z'], del_idx), :].reset_index(drop=True)

        if len(np.unique(T['Z'])) > 2:
            del_idx = np.array([])
            for k in np.unique(T['Z']):
                if sum(T['Z'] == k) < 0.01 * len(T['Z']):
                    del_idx = np.append(del_idx, k)
            T = T.loc[~np.in1d(T['Z'], del_idx), :].reset_index(drop=True)

        S_nPerClass = math.ceil(min(np.unique(S['Z'], return_counts=True)[
                                        1]))  # number of observations kept referenced by the min number of available observation per class
        T_nPerClass = math.ceil(min(np.unique(T['Z'], return_counts=True)[1]))

        z_kept_source = np.array([]).astype(int)
        for lab in np.unique(S['Z']):
            z_kept_source = np.append(z_kept_source,
                                      np.random.choice(np.where(S['Z'] == lab)[0], S_nPerClass, replace=False))

        S = S.loc[z_kept_source, :].reset_index(drop=True)

        z_kept_target = np.array([]).astype(int)
        for lab in np.unique(T['Z']):
            z_kept_target = np.append(z_kept_target,
                                      np.random.choice(np.where(T['Z'] == lab)[0], T_nPerClass, replace=False))
        T = T.loc[z_kept_target, :].reset_index(drop=True)

    # print(np.unique(S['Z'],return_counts=True))
    # print(np.unique(T['Z'],return_counts=True))

    if type_supervision == 'unsupervised':
        prop_S = 1
        prop_T = 0
        if alpha == None:
            if Objective_Variable == 'discrete':
                alpha = 0.661
            else:
                alpha = 0.3
    elif type_supervision == 'semi-supervised' and Labelled_Proportion_Target != None:
        prop_S = 1
        prop_T = Labelled_Proportion_Target
        if alpha == None:
            if Objective_Variable == 'discrete':
                alpha = 3.335
            else:
                alpha = 2.625
    elif (
            type_supervision == 'partial' or type_supervision == 'cross-partial') and Labelled_Proportion_Source != None and Labelled_Proportion_Target != None:
        prop_S = Labelled_Proportion_Source
        prop_T = Labelled_Proportion_Target
        if alpha == None:
            if Objective_Variable == 'discrete':
                alpha = 2.875
            else:
                alpha = 2.425
    else:
        return -1

    if Objective_Variable == 'discrete' or Objective_Variable == 'both':
        z_labelled_source = np.array([]).astype(int)
        for lab in np.unique(S['Z']):
            a = np.random.choice(np.where(S['Z'] == lab)[0], math.ceil(prop_S * sum(S['Z'] == lab)), replace=False)
            z_labelled_source = np.append(z_labelled_source, a)

        z_labelled_target = np.array([]).astype(int)
        for lab in np.unique(T['Z']):
            b = np.random.choice(np.where(T['Z'] == lab)[0], math.ceil(prop_T * sum(T['Z'] == lab)), replace=False)
            z_labelled_target = np.append(z_labelled_target, b)

        Z_training_data_source = S.loc[:, S.columns != 'Y']
        Z_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'] = -1

        Z_training_data_Target = T.loc[:, T.columns != 'Y']
        Z_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] = -1

    if Objective_Variable == 'both':
        # Memes indexs labellisés
        y_labelled_source = z_labelled_source
        y_labelled_target = z_labelled_target
        Y_training_data_source = S.loc[:, S.columns != 'Z']
        Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y'] = np.NaN
        Y_training_data_Target = T.loc[:, T.columns != 'Z']
        Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y'] = np.NaN

    if Objective_Variable == 'continuous':
        y_labelled_source = np.random.choice(np.arange(len(S['Y'])), math.ceil(prop_S * len(S['Y'])), replace=False)
        y_labelled_target = np.random.choice(np.arange(len(T['Y'])), math.ceil(prop_T * len(T['Y'])), replace=False)
        Y_training_data_source = S.loc[:, S.columns != 'Z']
        Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y'] = np.NaN
        Y_training_data_Target = T.loc[:, T.columns != 'Z']
        Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y'] = np.NaN

    perf_coot = np.array([])
    perf_jdcoot = np.array([])
    perf_coot_test = np.array([])
    perf_jdcoot_test = np.array([])
    perf_ref = np.array([])
    perf_ref2 = np.array([])

    ####reference perf

    if type_supervision == 'semi-supervised' and prop_T != 0 and prop_T != 1:

        if Objective_Variable == 'discrete' or Objective_Variable == 'both':
            def clf_seq(shape, nClass):
                model = tf_keras.Sequential([
                    Dense(units=128, input_shape=shape, activation='sigmoid'),
                    Dense(units=nClass, activation='sigmoid')])
                return model

            vfunc = np.vectorize(lambda arr: 'X' in arr)
            fe_size = sum(vfunc(T.columns))  # Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
            # loss = 'MeanSquaredError
            clf = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
            clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
            enc = onehot(handle_unknown='ignore',
                         categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
            # print(enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)))

            clf.fit(
                Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, Z_training_data_Target.columns != 'Z'],
                enc.fit_transform(
                    Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, 'Z'].values.reshape(-1, 1)),
                batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:, vfunc(T.columns)])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            perf_ref = np.append(perf_ref, sum(z_test == T_test.loc[:, 'Z']) / len(z_test))

            z_test = clf.predict(
                Z_training_data_Target.loc[Z_training_data_Target['Z'] == -1, Z_training_data_Target.columns != 'Z'])
            z_test = enc.inverse_transform(z_test).reshape(-1)
            perf_ref2 = np.append(perf_ref2, sum(z_test == T.loc[
                np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) / len(z_test))

        if Objective_Variable == 'continuous' or Objective_Variable == 'both':
            def clf_seq(shape, nClass):
                model = tf_keras.Sequential([
                    Dense(units=128, input_shape=shape, activation='linear'),
                    Dense(units=nClass, activation='linear')])
                return model

            vfunc = np.vectorize(lambda arr: 'X' in arr)
            fe_size = sum(vfunc(T.columns))  # Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'MeanSquaredError'
            clf = clf_seq(shape, nClass=1)
            clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

            clf.fit(Y_training_data_Target.loc[
                        ~np.isnan(Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y'],
                    Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']), 'Y'], batch_size=10, epochs=20,
                    verbose=0)  # we train the classifier with target data estimated
            # print(len(Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']),'Y']))

            z_test = clf.predict(T_test.loc[:, vfunc(T.columns)]).reshape(-1)
            perf_ref = np.append(perf_ref, sum((z_test - T_test.loc[:, 'Y']) ** 2) / len(z_test))

            z_test = clf.predict(Y_training_data_Target.loc[np.isnan(
                Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y']).reshape(-1)
            perf_ref2 = np.append(perf_ref2, sum((z_test - T.loc[
                np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y']) ** 2) / len(z_test))
            # print(perf_ref2)

    elif prop_T == 0:
        if Objective_Variable == 'discrete' or Objective_Variable == 'both':
            perf_ref = np.append(perf_ref, np.NaN)
            perf_ref2 = np.append(perf_ref2, np.NaN)
        if Objective_Variable == 'continuous' or Objective_Variable == 'both':
            perf_ref = np.append(perf_ref, np.NaN)
            perf_ref2 = np.append(perf_ref2, np.NaN)

    elif type_supervision == 'partial' or type_supervision == 'cross-partial':

        if Objective_Variable == 'discrete' or Objective_Variable == 'both':
            def clf_seq(shape, nClass):
                model = tf_keras.Sequential([
                    Dense(units=128, input_shape=shape, activation='sigmoid'),
                    Dense(units=nClass, activation='sigmoid')])
                return model

            vfunc = np.vectorize(lambda arr: 'X' in arr)
            fe_size = sum(vfunc(T.columns))  # Nombre de variables de Target
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
            # loss = 'MeanSquaredError
            clf = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
            clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
            enc = onehot(handle_unknown='ignore', sparse=False,
                         categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
            # print(enc.fit_transform(Z_training_data_Target.loc[Z_training_data_Target['Z']!=-1,'Z'].values.reshape(-1,1)))

            clf.fit(
                Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, Z_training_data_Target.columns != 'Z'],
                enc.fit_transform(
                    Z_training_data_Target.loc[Z_training_data_Target['Z'] != -1, 'Z'].values.reshape(-1, 1)),
                batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:, vfunc(T.columns)])
            z_test = enc.inverse_transform(z_test).reshape(-1)

            fe_size = sum(vfunc(S.columns))  # Nombre de variables de Targe
            shape = (fe_size,)
            loss = 'categorical_crossentropy'
            # loss = 'MeanSquaredError
            clf2 = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))

            clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

            clf2.fit(
                Z_training_data_source.loc[Z_training_data_source['Z'] != -1, Z_training_data_source.columns != 'Z'],
                enc.fit_transform(
                    Z_training_data_source.loc[Z_training_data_source['Z'] != -1, 'Z'].values.reshape(-1, 1)),
                batch_size=10, epochs=20, verbose=0)  # we train the classifier with target data estimated
            z_test2 = clf2.predict(S_test.loc[:, vfunc(S.columns)])
            z_test2 = enc.inverse_transform(z_test2).reshape(-1)

            perf_ref = np.append(perf_ref, (sum(z_test == T_test.loc[:, 'Z']) + sum(z_test2 == S_test.loc[:, 'Z'])) / (
                        len(z_test) + len(z_test2)))

            z_test = clf.predict(
                Z_training_data_Target.loc[Z_training_data_Target['Z'] == -1, Z_training_data_Target.columns != 'Z'])
            z_test = enc.inverse_transform(z_test).reshape(-1)

            z_test2 = clf2.predict(
                Z_training_data_source.loc[Z_training_data_source['Z'] == -1, Z_training_data_source.columns != 'Z'])
            z_test2 = enc.inverse_transform(z_test2).reshape(-1)

            perf_ref2 = np.append(perf_ref2, (
                        sum(z_test == T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) + sum(
                    z_test2 == S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'])) / (
                                              len(z_test) + len(z_test2)))

        if Objective_Variable == 'continuous' or Objective_Variable == 'both':
            def clf_seq(shape, nClass):
                model = tf_keras.Sequential([
                    Dense(units=128, input_shape=shape, activation='linear'),
                    Dense(units=nClass, activation='linear')])
                return model

            vfunc = np.vectorize(lambda arr: 'X' in arr)
            fe_size = sum(vfunc(T.columns))  # Nombre de variables de Targe
            shape = (fe_size,)
            # loss = 'categorical_crossentropy'
            loss = 'MeanSquaredError'
            clf = clf_seq(shape, nClass=1)
            clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

            clf.fit(Y_training_data_Target.loc[
                        ~np.isnan(Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y'],
                    Y_training_data_Target.loc[~np.isnan(Y_training_data_Target['Y']), 'Y'], batch_size=10, epochs=20,
                    verbose=0)  # we train the classifier with target data estimated

            z_test = clf.predict(T_test.loc[:, vfunc(T.columns)]).reshape(-1)

            fe_size = sum(vfunc(S.columns))  # Nombre de variables de Targe
            shape = (fe_size,)
            # loss = 'categorical_crossentropy'
            loss = 'MeanSquaredError'
            clf2 = clf_seq(shape, nClass=1)
            clf2.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

            clf2.fit(Y_training_data_source.loc[
                         ~np.isnan(Y_training_data_source['Y']), Y_training_data_source.columns != 'Y'],
                     Y_training_data_source.loc[~np.isnan(Y_training_data_source['Y']), 'Y'], batch_size=10, epochs=20,
                     verbose=0)  # we train the classifier with target data estimated

            z_test2 = clf2.predict(S_test.loc[:, vfunc(S.columns)]).reshape(-1)

            perf_ref = np.append(perf_ref, (
                        sum((z_test - T_test.loc[:, 'Y']) ** 2) + sum((z_test2 - S_test.loc[:, 'Y']) ** 2)) / (
                                             len(z_test) + len(z_test2)))

            z_test = clf.predict(Y_training_data_Target.loc[np.isnan(
                Y_training_data_Target['Y']), Y_training_data_Target.columns != 'Y']).reshape(-1)
            z_test2 = clf2.predict(Y_training_data_source.loc[np.isnan(
                Y_training_data_source['Y']), Y_training_data_source.columns != 'Y']).reshape(-1)

            perf_ref2 = np.append(perf_ref2, (sum((z_test - T.loc[
                np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Z']) ** 2) + sum(
                (z_test2 - S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Z']) ** 2)) / (
                                              len(z_test) + len(z_test2)))

    #####

    if Objective_Variable == 'both' or Objective_Variable == 'discrete':
        #####COOT######
        ###### DISCRETE ######
        if algo == 'both' or algo == 'COOT':

            # cost matrix with ot dist
            def compute_cost_matrix(ys, yt, v=10000):
                M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1), metric=comp_(v))
                return M

            if prop_T == 0 or type_supervision == 'semi-supervised':

                if prop_T == 0:
                    M_lin = None
                else:
                    M_lin = compute_cost_matrix(yt=Z_training_data_Target['Z'], ys=Z_training_data_source['Z'])

                Ts, Tv, cost = cot_numpy(X1=Z_training_data_source.loc[:, Z_training_data_source.columns != 'Z'],
                                         X2=Z_training_data_Target.loc[:, Z_training_data_Target.columns != 'Z'],
                                         niter=100, C_lin=M_lin,
                                         algo='sinkhorn', reg=1,
                                         algo2='emd', verbose=False)

                # Target estimation
                enc = onehot(handle_unknown='ignore', sparse=False,
                             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
                zs_onehot = enc.fit_transform(S['Z'].values.reshape(-1, 1))
                zt_onehot_estimated = len(T.loc[:, 'Z']) * np.dot(Ts.T, zs_onehot)
                zt_estimated = enc.inverse_transform(zt_onehot_estimated).reshape(-1)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0:
                    perf_coot = np.append(perf_coot, sum(
                        T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] == zt_estimated[
                            np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)]) / len(
                        np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)))
                    # perf_tot=sum(T.loc[:,'Z']==zt_estimated)/len(zt_estimated)
                else:
                    perf_coot = np.append(perf_coot, sum(T.loc[:, 'Z'] == zt_estimated) / len(zt_estimated))

                #############train classifier and evaluate performance on test
                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='sigmoid'),
                        Dense(units=nClass, activation='sigmoid')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_size = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_size,)
                loss = 'categorical_crossentropy'
                # loss = 'MeanSquaredError'
                # clf = clf_seq(shape, nClass = len(np.unique(S['Z'])))
                clf = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                # print(enc.fit_transform(zt_estimated.reshape(-1,1)))
                # print(len(np.union1d(np.unique(S['Z']),np.unique(T['Z']))))

                clf.fit(T.loc[:, vfunc(T.columns)], enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
                        epochs=20, verbose=0)  # we train the classifier with target data estimated
                z_test = clf.predict(T_test.loc[:, vfunc(T.columns)])
                z_test = enc.inverse_transform(z_test).reshape(-1)
                perf_coot_test = np.append(perf_coot_test, sum(z_test == T_test.loc[:, 'Z']) / len(z_test))




            ######################## 

            elif type_supervision == 'partial' or type_supervision == 'cross-partial':

                # Source labelled data learning
                Source_indexes_labelled = np.where(Z_training_data_source['Z'] != -1)[0]

                M_lin = compute_cost_matrix(yt=Z_training_data_Target['Z'],
                                            ys=Z_training_data_source.loc[Source_indexes_labelled, 'Z'])
                Ts, Tv, cost = cot_numpy(
                    X1=Z_training_data_source.loc[Source_indexes_labelled, Z_training_data_source.columns != 'Z'],
                    X2=Z_training_data_Target.loc[:, Z_training_data_Target.columns != 'Z'],
                    niter=100, C_lin=M_lin,
                    algo='sinkhorn', reg=1,
                    algo2='emd', verbose=False)
                # Target estimation
                enc = onehot(handle_unknown='ignore', sparse=False,
                             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
                zs_onehot = enc.fit_transform(S.loc[Source_indexes_labelled, 'Z'].values.reshape(-1, 1))
                zt_onehot_estimated = len(T.loc[:, 'Z']) * np.dot(Ts.T, zs_onehot)
                zt_estimated = enc.inverse_transform(zt_onehot_estimated).reshape(-1)

                # Target labelled data learning
                Target_indexes_labelled = np.where(Z_training_data_Target['Z'] != -1)[0]
                M_lin = compute_cost_matrix(yt=Z_training_data_source['Z'],
                                            ys=Z_training_data_Target.loc[Target_indexes_labelled, 'Z'])
                Ts, Tv, cost = cot_numpy(
                    X1=Z_training_data_Target.loc[Target_indexes_labelled, Z_training_data_Target.columns != 'Z'],
                    X2=Z_training_data_source.loc[:, Z_training_data_source.columns != 'Z'],
                    niter=100, C_lin=M_lin,
                    algo='sinkhorn', reg=1,
                    algo2='emd', verbose=False)
                # Source estimation
                enc = onehot(handle_unknown='ignore', sparse=False,
                             categories=[np.arange(len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))])
                zt_onehot = enc.fit_transform(T.loc[Target_indexes_labelled, 'Z'].values.reshape(-1, 1))
                zs_onehot_estimated = len(S.loc[:, 'Z']) * np.dot(Ts.T, zt_onehot)
                zs_estimated = enc.inverse_transform(zs_onehot_estimated).reshape(-1)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0 and len(
                        np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source)) != 0:
                    perf_coot = np.append(perf_coot, (sum(
                        T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'] == zt_estimated[
                            np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)]) + sum(
                        S.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z'] == zs_estimated[
                            np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source)])) / (
                                                      len(np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                       z_labelled_target)) + len(
                                                  np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source))))
                else:
                    perf_coot = np.append(perf_coot,
                                          (sum(T.loc[:, 'Z'] == zt_estimated) + sum(S.loc[:, 'Z'] == zs_estimated)) / (
                                                      len(zt_estimated) + len(zs_estimated)))

                #############train classifier and evaluate the performance on test
                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='sigmoid'),
                        Dense(units=nClass, activation='sigmoid')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeT = sum(vfunc(T.columns))  # Nombre de variables de Target
                shapeT = (fe_sizeT,)
                loss = 'categorical_crossentropy'
                clfT = clf_seq(shapeT, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeS = sum(vfunc(S.columns))  # Nombre de variables de Target
                shapeS = (fe_sizeS,)
                loss = 'categorical_crossentropy'
                clfS = clf_seq(shapeS, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                clfT.fit(T.loc[:, vfunc(T.columns)], enc.fit_transform(zt_estimated.reshape(-1, 1)), batch_size=10,
                         epochs=20, verbose=0)  # we train the classifier with target data estimated
                clfS.fit(S.loc[:, vfunc(S.columns)], enc.fit_transform(zs_estimated.reshape(-1, 1)), batch_size=10,
                         epochs=20, verbose=0)  # we train the classifier with target data estimated

                zt_test = clfT.predict(T_test.loc[:, vfunc(T_test.columns)])
                zs_test = clfS.predict(S_test.loc[:, vfunc(S_test.columns)])

                zt_test = enc.inverse_transform(zt_test).reshape(-1)
                zs_test = enc.inverse_transform(zs_test).reshape(-1)

                perf_coot_test = np.append(perf_coot_test,
                                           (sum(zt_test == T_test.loc[:, 'Z']) + sum(zs_test == S_test.loc[:, 'Z'])) / (
                                                       len(zs_test) + len(zt_test)))
            ######################## 

        ######JDCOT######
        ###### DISCRETE ######

        if algo == 'both' or algo == 'JDCOOT':

            if prop_T >= 0 and type_supervision != "cross-partial":

                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='relu'),
                        Dense(units=nClass, activation='softmax')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_sizeB,)
                loss = 'categorical_crossentropy'
                # loss = 'MeanSquaredError'
                clfB = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
                shape = (fe_sizeA,)
                loss = 'categorical_crossentropy'
                # loss = 'MeanSquaredError'
                clfA = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                def one_hot(y, nClass):

                    m = min(y)
                    if m == -1:
                        if len(np.unique(y)) != 1:
                            m = np.sort(np.unique(y))[1]

                    Y = np.zeros((len(y), nClass))
                    for i in range(len(y)):
                        if y[i] != -1:
                            Y[i, (y[i] - m).astype(int)] = 1
                    return Y

                def one_hot_inv(z_encoded):
                    return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))

                oh_source = one_hot(Z_training_data_source.loc[:, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                oh_target = one_hot(Z_training_data_Target.loc[:, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                # model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                model1, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                                  XA=np.array(Z_training_data_source.loc[:,
                                                                              Z_training_data_source.columns != 'Z']),
                                                                  YA=oh_source,
                                                                  XB=np.array(Z_training_data_Target.loc[:,
                                                                              Z_training_data_Target.columns != 'Z']),
                                                                  YB=oh_target,
                                                                  yAtruth=S['Z'],
                                                                  yBtruth=T['Z'], algo='sinkhorn', reg=1, alpha=alpha)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0:

                    zpred_enc_target = model2.predict(Z_training_data_Target.loc[
                                                          np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                       z_labelled_target), Z_training_data_Target.columns != 'Z'])
                    zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(T['Z']))

                    if type_supervision != 'partial':
                        perf_jdcoot = np.append(perf_jdcoot, sum(zpred_target == T.loc[
                            np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z']) / len(zpred_target))
                        zt_test = one_hot_inv(model2.predict(T_test.loc[:, vfunc(T_test.columns)])) + min(
                            np.unique(T['Z']))
                        perf_jdcoot_test = np.append(perf_jdcoot_test,
                                                     (sum(zt_test == T_test.loc[:, 'Z'])) / (len(zt_test)))

                    # print(zpred_target)
                    # print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),z_labelled_target),'Z'])
                    if type_supervision == 'partial':
                        zpred_enc_source = model1.predict(Z_training_data_source.loc[
                                                              np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                                           z_labelled_source), Z_training_data_source.columns != 'Z'])
                        zpred_source = one_hot_inv(zpred_enc_source) + min(np.unique(S['Z']))

                        perf_jdcoot = np.append(perf_jdcoot, (sum(zpred_source == S.loc[
                            np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z']) + sum(
                            zpred_target == T.loc[
                                np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'])) / (
                                                            len(zpred_source) + len(zpred_target)))
                        zt_test = one_hot_inv(model2.predict(T_test.loc[:, vfunc(T_test.columns)])) + min(
                            np.unique(T['Z']))
                        zs_test = one_hot_inv(model1.predict(S_test.loc[:, vfunc(S_test.columns)])) + min(
                            np.unique(T['Z']))
                        perf_jdcoot_test = np.append(perf_jdcoot_test, (
                                    sum(zt_test == T_test.loc[:, 'Z']) + sum(zs_test == S_test.loc[:, 'Z'])) / (
                                                                 len(zt_test) + len(zs_test)))

                else:
                    print("Set automatically to 1")
                    perf_jdcoot = np.append(perf_jdcoot, 1)

            ############################################################
            elif type_supervision == 'cross-partial':

                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='relu'),
                        Dense(units=nClass, activation='softmax')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_sizeB,)
                loss = 'categorical_crossentropy'
                # loss = 'MeanSquaredError'
                clfB = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
                shape = (fe_sizeA,)
                loss = 'categorical_crossentropy'
                # loss = 'MeanSquaredError'
                clfA = clf_seq(shape, nClass=len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                def one_hot(y, nClass):
                    y = np.array(y)
                    m = min(y)
                    if m == -1:
                        if len(np.unique(y)) != 1:
                            m = np.sort(np.unique(y))[1]

                    Y = np.zeros((len(y), nClass))
                    for i in range(len(y)):
                        if y[i] != -1:
                            Y[i, (y[i] - m).astype(int)] = 1
                    return Y

                def one_hot_inv(z_encoded):
                    return np.vectorize(lambda i: np.argmax(z_encoded[i, :]))(np.arange(z_encoded.shape[0]))

                # Source Model estimation
                oh_source = one_hot(Z_training_data_source.loc[:, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                # no unlabelled anymore
                oh_target = one_hot(Z_training_data_Target.loc[Z_training_data_Target.loc[:, 'Z'] != -1, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                # model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                mod, model1, results = jdcot_multitask_classif(modelB=clfA, modelA=clfB,
                                                               XB=np.array(Z_training_data_source.loc[:,
                                                                           Z_training_data_source.columns != 'Z']),
                                                               YB=oh_source,
                                                               XA=np.array(Z_training_data_Target.loc[
                                                                               Z_training_data_Target.loc[:,
                                                                               'Z'] != -1, Z_training_data_Target.columns != 'Z']),
                                                               YA=oh_target,
                                                               yBtruth=S['Z'],
                                                               yAtruth=T.loc[
                                                                   Z_training_data_Target.loc[:, 'Z'] != -1, 'Z'],
                                                               algo='sinkhorn', reg=1, alpha=alpha)

                # Target Model estimation
                oh_source = one_hot(Z_training_data_source.loc[Z_training_data_source.loc[:, 'Z'] != -1, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                # no unlabelled anymore
                oh_target = one_hot(Z_training_data_Target.loc[:, 'Z'],
                                    len(np.union1d(np.unique(S['Z']), np.unique(T['Z']))))
                # model1,model2,results=jdcot_multitask_classif(clfB,clfA,XA,YA,XB,YB,yAtruth,yBtruth)

                mod, model2, results = jdcot_multitask_classif(modelA=clfA, modelB=clfB,
                                                               XA=np.array(Z_training_data_source.loc[
                                                                               Z_training_data_source.loc[:,
                                                                               'Z'] != -1, Z_training_data_source.columns != 'Z']),
                                                               YA=oh_source,
                                                               XB=np.array(Z_training_data_Target.loc[:,
                                                                           Z_training_data_Target.columns != 'Z']),
                                                               YB=oh_target,
                                                               yAtruth=S.loc[
                                                                   Z_training_data_source.loc[:, 'Z'] != -1, 'Z'],
                                                               yBtruth=T['Z'], algo='sinkhorn', reg=1, alpha=alpha)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target)) != 0:

                    zpred_enc_target = model2.predict(Z_training_data_Target.loc[
                                                          np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                       z_labelled_target), Z_training_data_Target.columns != 'Z'])
                    zpred_target = one_hot_inv(zpred_enc_target) + min(np.unique(T['Z']))

                    zpred_enc_source = model1.predict(Z_training_data_source.loc[
                                                          np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                                       z_labelled_source), Z_training_data_source.columns != 'Z'])
                    zpred_source = one_hot_inv(zpred_enc_source) + min(np.unique(S['Z']))

                    perf_jdcoot = np.append(perf_jdcoot, (sum(zpred_source == S.loc[
                        np.setdiff1d(np.arange(0, np.shape(S)[0]), z_labelled_source), 'Z']) + sum(
                        zpred_target == T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]), z_labelled_target), 'Z'])) / (
                                                        len(zpred_source) + len(zpred_target)))
                    zt_test = one_hot_inv(model2.predict(T_test.loc[:, vfunc(T_test.columns)])) + min(np.unique(T['Z']))
                    zs_test = one_hot_inv(model1.predict(S_test.loc[:, vfunc(S_test.columns)])) + min(np.unique(T['Z']))
                    perf_jdcoot_test = np.append(perf_jdcoot_test, (
                                sum(zt_test == T_test.loc[:, 'Z']) + sum(zs_test == S_test.loc[:, 'Z'])) / (
                                                             len(zt_test) + len(zs_test)))

                else:
                    print("Set automatically to 1")
                    perf_jdcoot = np.append(perf_jdcoot, 1)

                    ############################################################               

    if Objective_Variable == 'both' or Objective_Variable == 'continuous':
        #####COOT######
        ###### CONTINUOUS ######
        if algo == 'both' or algo == 'COOT':

            # cost matrix with ot dist
            def compute_cost_matrix(ys, yt):
                M = ot.dist(ys.values.reshape(-1, 1), yt.values.reshape(-1, 1),
                            metric=comp_regression())  # comp_reg ? ou comp_
                return M

            if prop_T == 0 or type_supervision == 'semi-supervised':

                if prop_T == 0:
                    M_lin = None
                else:
                    M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'], ys=Y_training_data_source['Y'])

                    # plt.imshow(M_lin)

                Ts, Tv, cost = cot_numpy(X1=Y_training_data_source.loc[:, Y_training_data_source.columns != 'Y'],
                                         X2=Y_training_data_Target.loc[:, Y_training_data_Target.columns != 'Y'],
                                         niter=100, C_lin=M_lin,
                                         algo='sinkhorn', reg=1,
                                         algo2='emd', verbose=False)

                # Target estimation
                # PLUS DE OH ENC
                # Nombre de variables de Target
                zt_estimated = len(T.loc[:, 'Y']) * np.dot(Ts.T, S['Y'])
                # print(S['Y'])
                # print(np.dot(Ts.T,S['Y']))

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
                    # perfs_tot=np.append(perf_coot,sum((zt_estimated-T.loc[:,'Y'])**2)/len(zt_estimated)
                    perf_coot = np.append(perf_coot, sum((zt_estimated[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                                    y_labelled_target)] - T.loc[
                                                              np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                           y_labelled_target), 'Y']) ** 2) / len(
                        np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)))

                else:
                    perf_coot = np.append(perf_coot, sum((T.loc[:, 'Y'] - zt_estimated) ** 2) / len(zt_estimated))

                # print(zt_estimated[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target)])
                # print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])
                # print(zt_estimated[np.sort(y_labelled_target)])
                # print(T.loc[np.sort(y_labelled_target),'Y'])

                #############train classifier and evaluate the performance on test
                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='linear'),
                        Dense(units=nClass, activation='linear')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_size = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_size,)
                # loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clf = clf_seq(shape, nClass=1)
                clf.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])
                # print(T.loc[:,vfunc(T.columns)])
                clf.fit(T.loc[:, vfunc(T.columns)], zt_estimated, batch_size=10, epochs=20,
                        verbose=0)  # we train the classifier with target data estimated
                z_test = clf.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
                # print(z_test)
                # print(T_test.loc[:,'Y'])
                perf_coot_test = np.append(perf_coot_test, sum((z_test - T_test.loc[:, 'Y']) ** 2) / len(z_test))
            ######################## 

            if type_supervision == 'partial' or type_supervision == 'cross-partial':

                Source_indexes_labelled = np.where(~np.isnan(Y_training_data_source['Y']))[0]
                M_lin = compute_cost_matrix(yt=Y_training_data_Target['Y'],
                                            ys=Y_training_data_source.loc[Source_indexes_labelled, 'Y'])
                Ts, Tv, cost = cot_numpy(
                    X1=Y_training_data_source.loc[Source_indexes_labelled, Y_training_data_source.columns != 'Y'],
                    X2=Y_training_data_Target.loc[:, Y_training_data_Target.columns != 'Y'],
                    niter=100, C_lin=M_lin,
                    algo='sinkhorn', reg=1,
                    algo2='emd', verbose=False)
                # Target estimation
                # PLUS DE OH ENC
                zt_estimated = len(T.loc[:, 'Y']) * np.dot(Ts.T, S.loc[Source_indexes_labelled, 'Y'])

                # Target labelled data learning
                Target_indexes_labelled = np.where(~np.isnan(Y_training_data_Target['Y']))[0]
                M_lin = compute_cost_matrix(yt=Y_training_data_source['Y'],
                                            ys=Y_training_data_Target.loc[Target_indexes_labelled, 'Y'])
                Ts, Tv, cost = cot_numpy(
                    X1=Y_training_data_Target.loc[Target_indexes_labelled, Y_training_data_Target.columns != 'Y'],
                    X2=Y_training_data_source.loc[:, Y_training_data_source.columns != 'Y'],
                    niter=100, C_lin=M_lin,
                    algo='sinkhorn', reg=1,
                    algo2='emd', verbose=False)
                # Source estimation
                # PLUS DE OH ENC
                zs_estimated = len(S.loc[:, 'Y']) * np.dot(Ts.T, T.loc[Target_indexes_labelled, 'Y'])

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0 and len(
                        np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source)) != 0:
                    perf_coot = np.append(perf_coot, (sum((T.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                              y_labelled_target), 'Y'] - zt_estimated[
                                                               np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                            y_labelled_target)]) ** 2) + sum((S.loc[
                                                                                                                  np.setdiff1d(
                                                                                                                      np.arange(
                                                                                                                          0,
                                                                                                                          np.shape(
                                                                                                                              S)[
                                                                                                                              0]),
                                                                                                                      y_labelled_source), 'Y'] -
                                                                                                              zs_estimated[
                                                                                                                  np.setdiff1d(
                                                                                                                      np.arange(
                                                                                                                          0,
                                                                                                                          np.shape(
                                                                                                                              S)[
                                                                                                                              0]),
                                                                                                                      y_labelled_source)]) ** 2)) / (
                                                      len(np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                       y_labelled_target)) + len(
                                                  np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source))))
                else:
                    perf_coot = np.append(perf_coot, (
                                sum((T.loc[:, 'Y'] - zt_estimated) ** 2) + sum((S.loc[:, 'Y'] - zs_estimated) ** 2)) / (
                                                      len(zt_estimated) + len(zs_estimated)))
                    # perf_coot=np.append(perf_coot,1)

                #############train classifier and evaluate the performance on test
                def clf_seq(shape, nClass):
                    model = tf_keras.Sequential([
                        Dense(units=128, input_shape=shape, activation='linear'),
                        Dense(units=nClass, activation='linear')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeT = sum(vfunc(T.columns))  # Nombre de variables de Target
                shapeT = (fe_sizeT,)
                loss = 'MeanSquaredError'
                clfT = clf_seq(shapeT, nClass=1)
                clfT.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeS = sum(vfunc(S.columns))  # Nombre de variables de Target
                shapeS = (fe_sizeS,)
                loss = 'MeanSquaredError'
                clfS = clf_seq(shapeS, nClass=1)
                clfS.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                clfT.fit(T.loc[:, vfunc(T.columns)], zt_estimated, batch_size=10, epochs=20,
                         verbose=0)  # we train the classifier with target data estimated
                clfS.fit(S.loc[:, vfunc(S.columns)], zs_estimated, batch_size=10, epochs=20,
                         verbose=0)  # we train the classifier with target data estimated

                zt_test = clfT.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
                zs_test = clfS.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]
                # print(len(zt_test))
                # print(len(zs_test))

                perf_coot_test = np.append(perf_coot_test, (
                            sum((zt_test - T_test.loc[:, 'Y']) ** 2) + sum((zs_test - S_test.loc[:, 'Y']) ** 2)) / (
                                                       len(zs_test) + len(zt_test)))
            ######################## 

        #####CONTINUOUS######
        if algo == 'both' or algo == 'JDCOOT':

            if prop_T >= 0 and type_supervision != "cross-partial":  # all cases

                """
                def clf_seq(shape,nClass):
                    model = tf_keras.Sequential([
                    Dense(units=128,input_shape=shape,activation='relu'),
                    Dense(units=nClass,activation = 'softmax')        ])
                    return model
                """

                def clf_seq(shape):
                    model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                                 Dense(units=1, activation='linear')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_sizeB,)
                # loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfB = clf_seq(shape)
                clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
                shape = (fe_sizeA,)
                # loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfA = clf_seq(shape)
                clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                # Y_training_data_source.loc[Y_training_data_source['Y']==-1,'Y']=np.NaN
                # Y_training_data_Target.loc[Y_training_data_Target['Y']==-1,'Y']=np.NaN

                """
                jdcot multi-task for multi regression problems
                npreds : number of parameters to predict (it has to be the same number for both datasets)
                yAtruth is (nA,npreds),yBtruth is (nB,npreds) : true value of the parameters to estimate
                YA is (nA,npreds), YB is (nB,npreds) : line of 0 if non observed labels and true values if observed labels (semi supervision)
                """
                # model1,model2,results=jdcot_multitask_reg(modelA,modelB,XA,YA,XB,YB,yAtruth,yBtruth)
                model1, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                                              XA=np.array(Y_training_data_source.loc[:,
                                                                          Y_training_data_source.columns != 'Y']),
                                                              YA=np.array(Y_training_data_source['Y']).reshape((-1, 1)),
                                                              XB=np.array(Y_training_data_Target.loc[:,
                                                                          Y_training_data_Target.columns != 'Y']),
                                                              YB=np.array(Y_training_data_Target['Y']).reshape((-1, 1)),
                                                              yAtruth=S['Y'],
                                                              yBtruth=T['Y'], reshape_data=False, algo='sinkhorn',
                                                              reg=100, alpha=alpha)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
                    zpred_target = model2.predict(Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                                          y_labelled_target), Y_training_data_Target.columns != 'Y'])
                    zpred_target = zpred_target.ravel()

                    if type_supervision != 'partial':
                        perf_jdcoot = np.append(perf_jdcoot, sum((zpred_target - T.loc[
                            np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target), 'Y']) ** 2) / len(
                            zpred_target))

                    # print(zpred_target)
                    # print(T.loc[np.setdiff1d(np.arange(0,np.shape(T)[0]),y_labelled_target),'Y'])
                    if type_supervision == 'partial':
                        zpred_source = model1.predict(Y_training_data_source.loc[
                                                          np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                                       y_labelled_source), Y_training_data_source.columns != 'Y'])
                        zpred_source = zpred_source.ravel()
                        perf_jdcoot = np.append(perf_jdcoot, (sum((zpred_source - S.loc[
                            np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y']) ** 2) + sum((
                                                                                                                         zpred_target -
                                                                                                                         T.loc[
                                                                                                                             np.setdiff1d(
                                                                                                                                 np.arange(
                                                                                                                                     0,
                                                                                                                                     np.shape(
                                                                                                                                         T)[
                                                                                                                                         0]),
                                                                                                                                 y_labelled_target), 'Y']) ** 2)) / (
                                                            len(zpred_source) + len(zpred_target)))
                else:
                    print("Set automatically to 1")
                    perf_jdcoot = np.append(perf_jdcoot, 1)

                ###ON TEST
                zt_test = model2.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
                zs_test = model1.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]
                # print(zs_test)
                # print(sum((zs_test-S_test.loc[:,'Z'])**2)/len(zs_test))
                if type_supervision == 'partial':
                    perf_jdcoot_test = np.append(perf_jdcoot_test, (
                                sum((zt_test - T_test.loc[:, 'Y']) ** 2) + sum((zs_test - S_test.loc[:, 'Y']) ** 2)) / (
                                                             len(zt_test) + len(zs_test)))

                else:
                    perf_jdcoot_test = np.append(perf_jdcoot_test,
                                                 (sum((zt_test - T_test.loc[:, 'Y']) ** 2)) / (len(zt_test)))

            ############################################################
            elif type_supervision == 'cross-partial':

                def clf_seq(shape):
                    model = tf_keras.Sequential([Dense(units=128, input_shape=shape, activation='linear'),
                                                 Dense(units=1, activation='linear')])
                    return model

                vfunc = np.vectorize(lambda arr: 'X' in arr)
                fe_sizeB = sum(vfunc(T.columns))  # Nombre de variables de Target
                shape = (fe_sizeB,)
                # loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfB = clf_seq(shape)
                clfB.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                fe_sizeA = sum(vfunc(S.columns))  # Nombre de variables de Source
                shape = (fe_sizeA,)
                # loss = 'categorical_crossentropy'
                loss = 'MeanSquaredError'
                clfA = clf_seq(shape)
                clfA.compile(optimizer='Adam', loss=loss, metrics=['accuracy'])

                # Source model

                mod, model1, results = jdcot_multitask_reg(modelB=clfA, modelA=clfB,
                                                           XB=np.array(Y_training_data_source.loc[:,
                                                                       Y_training_data_source.columns != 'Y']),
                                                           YB=np.array(Y_training_data_source['Y']).reshape((-1, 1)),
                                                           XA=np.array(Y_training_data_Target.loc[~np.isnan(
                                                               Y_training_data_Target.loc[:,
                                                               'Y']), Y_training_data_Target.columns != 'Y']),
                                                           YA=np.array(Y_training_data_Target.loc[~np.isnan(
                                                               Y_training_data_Target.loc[:, 'Y']), 'Y']).reshape(
                                                               (-1, 1)),
                                                           yBtruth=S['Y'],
                                                           yAtruth=T.loc[
                                                               ~np.isnan(Y_training_data_Target.loc[:, 'Y']), 'Y'],
                                                           reshape_data=False, algo='sinkhorn', reg=100, alpha=alpha)
                # Target Model
                mod, model2, results = jdcot_multitask_reg(modelA=clfA, modelB=clfB,
                                                           XA=np.array(Y_training_data_source.loc[~np.isnan(
                                                               Y_training_data_source.loc[:,
                                                               'Y']), Y_training_data_source.columns != 'Y']),
                                                           YA=np.array(Y_training_data_source.loc[~np.isnan(
                                                               Y_training_data_source.loc[:, 'Y']), 'Y']).reshape(
                                                               (-1, 1)),
                                                           XB=np.array(Y_training_data_Target.loc[:,
                                                                       Y_training_data_Target.columns != 'Y']),
                                                           YB=np.array(Y_training_data_Target['Y']).reshape((-1, 1)),
                                                           yAtruth=S.loc[
                                                               ~np.isnan(Y_training_data_source.loc[:, 'Y']), 'Y'],
                                                           yBtruth=T['Y'], reshape_data=False, algo='sinkhorn', reg=100,
                                                           alpha=alpha)

                if len(np.setdiff1d(np.arange(0, np.shape(T)[0]), y_labelled_target)) != 0:
                    zpred_target = model2.predict(Y_training_data_Target.loc[np.setdiff1d(np.arange(0, np.shape(T)[0]),
                                                                                          y_labelled_target), Y_training_data_Target.columns != 'Y'])
                    zpred_target = zpred_target.ravel()

                    zpred_source = model1.predict(Y_training_data_source.loc[np.setdiff1d(np.arange(0, np.shape(S)[0]),
                                                                                          y_labelled_source), Y_training_data_source.columns != 'Y'])
                    zpred_source = zpred_source.ravel()
                    perf_jdcoot = np.append(perf_jdcoot, (sum((zpred_source - S.loc[
                        np.setdiff1d(np.arange(0, np.shape(S)[0]), y_labelled_source), 'Y']) ** 2) + sum((zpred_target -
                                                                                                          T.loc[
                                                                                                              np.setdiff1d(
                                                                                                                  np.arange(
                                                                                                                      0,
                                                                                                                      np.shape(
                                                                                                                          T)[
                                                                                                                          0]),
                                                                                                                  y_labelled_target), 'Y']) ** 2)) / (
                                                        len(zpred_source) + len(zpred_target)))
                else:
                    print("Set automatically to 1")
                    perf_jdcoot = np.append(perf_jdcoot, 1)

                ###ON TEST
                zt_test = model2.predict(T_test.loc[:, vfunc(T_test.columns)])[:, 0]
                zs_test = model1.predict(S_test.loc[:, vfunc(S_test.columns)])[:, 0]
                # print(zs_test)
                # print(sum((zs_test-S_test.loc[:,'Z'])**2)/len(zs_test))
                perf_jdcoot_test = np.append(perf_jdcoot_test, (
                            sum((zt_test - T_test.loc[:, 'Y']) ** 2) + sum((zs_test - S_test.loc[:, 'Y']) ** 2)) / (
                                                         len(zt_test) + len(zs_test)))

        ############################################################

    # print(perf_coot_test)
    # print(perf_jdcoot_test)
    if len(perf_coot) == 0: perf_coot = np.NaN
    if len(perf_jdcoot) == 0: perf_jdcoot = np.NaN
    if len(perf_coot_test) == 0: perf_coot_test = np.NaN
    if len(perf_jdcoot_test) == 0: perf_jdcoot_test = np.NaN

    print("Pure Performance COOT : {} ".format(perf_coot))
    print("Test Performance COOT : {} ".format(perf_coot_test))
    print("\n")
    print("Pure Performance JDCOOT : {} ".format(perf_jdcoot))
    print("Test Performance JDCOOT : {} ".format(perf_jdcoot_test))
    print("\n")
    print("Pure Performance Reference : {} ".format(perf_ref2))
    print("Test Performance Reference : {} ".format(perf_ref))

    return (perf_coot, perf_jdcoot, perf_coot_test, perf_jdcoot_test, perf_ref, perf_ref2)
