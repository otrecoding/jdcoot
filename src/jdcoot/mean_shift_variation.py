import math

import matplotlib.pyplot as plt
import numpy as np

from .data_generation import data_generator
from .globals import INDEX_GENERATION
from .graph_multi import graph_multi
from .performance import Performance
from .read_data import Read
from .reference_scenario import Sref
from .store_data import Store

# +

def Mean_Shift_Variation(Mean_Shift, Data=None, Monte_Carlo=1, Multi_Variations=True, algo='both',
                         type_supervision='unsupervised', Objective_Variable='discrete', Balance=True, alpha=None,
                         Poisson=False, Labelled_Proportion_Target=None,
                         Labelled_Proportion_Source=None
                         , n_Source=1000, n_Target=1000, d_Source=100, d_Target=100, mean_X_Source=np.zeros(100),
                         mean_X_Target=np.zeros(100),
                         mean_Y_Source=0, mean_Y_Target=0, rho_source_generation=0.7, rho_source_non_generation=0.2,
                         rho_target_generation=0.7, rho_target_non_generation=0.2, Sparse_Rate=0.75,
                         Odds_Ratio_Source=0.5, Odds_Ratio_Target=0.5, R2_Source=0.6, R2_Target=0.6,
                         Observed_Covariates_Proportion_Target=0.2,
                         Observed_Covariates_Proportion_Source=0.2, Indexes_Chosen_For_Generation=INDEX_GENERATION):
    """
    Mean_Shift_Variation : Compute and plot performance versus the mean shift between target and source

    Input : Mean_Shift (np array) : Mean Shift values between target and source to compute performance with

            Data (tuple or str) :
                                (str) Name of the folder where data has been stored

                                OR

                                Previous simulated data (tuple of 10) with respectively :
                                                            Pure performance Monte Carlo COOT,
                                                             Pure performance Monte Carlo JDCOOT,
                                                             Pure performance Monte Carlo REF,
                                                             Pure performance Boxplot COOT,
                                                             Pure performance Boxplot JDCOOT,
                                                             Test performance Monte Carlo COOT,
                                                             Test performance Monte Carlo JDCOOT,
                                                             Test performance Monte Carlo REF,
                                                             Test performance Boxplot COOT,
                                                             Test performance Boxplot JDCOOT



            type_supervision (str : 'unsupervised' or 'semi-supervised' or 'partial' or 'cross-partial') : if 'unsupervised', measure the performance such that none of the observations of target are labelled and all source observations are labelled
                                                                                                           if 'semi-supervised', measure the performance such that labelled proportions of observations of target vary
                                                                                                           if 'partial', measure the performance such that labelled proportions of observations of target and source vary
                                                                                                           if 'cross-partial', measure the performance such that labelled proportions of observations of target and source vary. Here, for JDCOOT, uses 2 semi-supervised JDCOOT methods for each dataset



            Monte_Carlo : Number of repetitions of computing


            Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps


            Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation
                                                                    if 'continuous', measure the performance for regression

            algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT
                                                        if 'JDCOOT', measure the performance for JDCOOT
                                                        if 'both',  measure the performance for COOT and JDCOOT

            Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data.

            alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())

            Poisson (Bool) : Uses Poisson modeling to generate more than 2 classes

            Labelled_Proportion_Target (float in [0,1]) : Proportion on observed labels in target for semi-supervised or partial analysis. Optional in case of unsupervised analysis.

            Labelled_Proportion_Source (float in [0,1]) : Proportion on observed labels in target for partial analysis. Optional in case of unsupervised or semi-supervised analysis.

            d_Source, d_Target (int): number of variables of source/target (must be the same for the generation in the "same world"/ consider that this set of variable is the "universe")

            mean_X_Source, mean_X_Target (d_Source/d_Target dimensional array of float) : mean of the Normal law of the covariates of source/target

            mean_Y_Source,mean_Y_Target (float) : mean of the continuous objective variable of source/target

            Source_Generation_Correlation, Source_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for source

            Target_Generation_Correlation, Target_Non_Generation_Correlation (float in  [0,1]): auto correlation coefficient of active/non active covariates for target

            Sparse_Rate (float in  [0,1]) : Proportion of active covariates for generation (Same for Source and Traget because generation in the "same world"/ consider that this generation explains the observed phenomenon)

            Odds_Ratio_Source, Odds_Ratio_Target (float) : Odds ratio of the model of source/target (for probabilities calculation in discrete case)

            R2_Source, R2_Target (float in  [0,1]) : R^2 of the model of source/target (for white noise calculation in continuous case)

            Observed_Covariates_Proportion_Source, Observed_Covariates_Proportion_Target (float in  [0,1]) : Proportion of observed covariates of source/target

            Indexes_Chosen_For_Generation (array of int) : Array of indexes of active variables. (In order to keep the same generation when we want to generate a test sample) If None, chosen randomly





    Output : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :
                                                                            Pure performance Monte Carlo COOT,
                                                                            Pure performance Monte Carlo JDCOOT,
                                                                            Pure performance Monte Carlo REF,
                                                                            Pure performance Boxplot COOT,
                                                                            Pure performance Boxplot JDCOOT,
                                                                            Pure performance Heatmap COOT,
                                                                            Pure performance Heatmap JDCOOT,
                                                                            Test performance Monte Carlo COOT,
                                                                            Test performance Monte Carlo JDCOOT,
                                                                            Test performance Monte Carlo REF,
                                                                            Test performance Boxplot COOT,
                                                                            Test performance Boxplot JDCOOT,
                                                                            Test performance Heatmap COOT,
                                                                            Test performance Heatmap JDCOOT,



                                                                        For Univariation Scenarios (No Heatmap) : 10 elements :
                                                                            Pure performance Monte Carlo COOT,
                                                                            Pure performance Monte Carlo JDCOOT,
                                                                            Pure performance Monte Carlo REF,
                                                                            Pure performance Boxplot COOT,
                                                                            Pure performance Boxplot JDCOOT,
                                                                            Test performance Monte Carlo COOT,
                                                                            Test performance Monte Carlo JDCOOT,
                                                                            Test performance Monte Carlo REF,
                                                                            Test performance Boxplot COOT,
                                                                            Test performance Boxplot JDCOOT

    """

    if Objective_Variable == 'both':
        return -1

    Sample_sizes = Mean_Shift
    R2 = Sample_sizes
    Store_in = True

    ######################### test samples generation : 
    ##
    ex = Sref()
    Col_observed_target = ex[1].columns[:-2]
    Col_observed_source = ex[0].columns[:-2]
    ex = None
    ##
    tests1 = []
    tests2 = []
    for samp_s in Sample_sizes:
        for samp_t in Sample_sizes:
            scenario_test = data_generator(math.ceil(n_Source * 0.3), math.ceil(n_Target * 0.3), d_Source, d_Target,
                                           mean_X_Source + samp_s, mean_X_Target + samp_t, mean_Y_Source,
                                           mean_Y_Target, rho_source_generation, rho_source_non_generation,
                                           rho_target_generation, rho_target_non_generation, Sparse_Rate,
                                           Odds_Ratio_Source=Odds_Ratio_Source, Odds_Ratio_Target=Odds_Ratio_Target,
                                           R2_Source=R2_Source, R2_Target=R2_Source,
                                           Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source,
                                           Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                           Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                           Cols_Chosen_For_Observation_Source=Col_observed_source,
                                           Cols_Chosen_For_Observation_Target=Col_observed_target, Poisson=Poisson)

            tests1.append(scenario_test)
            if samp_s == R2[0]:
                tests2.append(scenario_test)

        #####################""

    if Multi_Variations:

        if isinstance(Data, str):
            Data = Read(Data)
            Store_in = False

        if (type(Data) is tuple and len(Data) != 2):
            if len(Data) == 14:
                MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, hm_coot, hm_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot, test_hm_coot, test_hm_jdcoot = Data
                Store_in = False
            else:
                print("Wrong data format")
                return 0

        else:

            test_hm_coot = np.zeros((len(Sample_sizes), len(Sample_sizes)))
            test_hm_jdcoot = np.zeros((len(Sample_sizes), len(Sample_sizes)))
            hm_coot = np.zeros((len(Sample_sizes), len(Sample_sizes)))
            hm_jdcoot = np.zeros((len(Sample_sizes), len(Sample_sizes)))

            MC_coot_perf = np.repeat(0, len(Sample_sizes))
            MC_jdcoot_perf = np.repeat(0, len(Sample_sizes))
            bp_coot = np.array([[]])
            bp_jdcoot = np.array([[]])

            test_MC_coot_perf = np.repeat(0, len(Sample_sizes))
            test_MC_jdcoot_perf = np.repeat(0, len(Sample_sizes))
            test_bp_coot = np.array([[]])
            test_bp_jdcoot = np.array([[]])

            test_MC_ref_perf = np.repeat(0, len(Sample_sizes))
            MC_ref_perf = np.repeat(0, len(Sample_sizes))

            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot = np.array([])
                test_z_jdcoot = np.array([])
                z_coot = np.array([])
                z_jdcoot = np.array([])
                coot_perf = np.array([])
                jdcoot_perf = np.array([])
                test_coot_perf = np.array([])
                test_jdcoot_perf = np.array([])

                ref = np.array([])
                test_ref = np.array([])
                j = 0
                for samp_s in Sample_sizes:
                    for samp_t in Sample_sizes:
                        scenario = data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source + samp_s,
                                                  mean_X_Target + samp_t, mean_Y_Source,
                                                  mean_Y_Target, rho_source_generation, rho_source_non_generation,
                                                  rho_target_generation, rho_target_non_generation, Sparse_Rate,
                                                  Odds_Ratio_Source=Odds_Ratio_Source,
                                                  Odds_Ratio_Target=Odds_Ratio_Target, R2_Source=R2_Source,
                                                  R2_Target=R2_Source,
                                                  Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source,
                                                  Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                                  Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                                  Cols_Chosen_For_Observation_Source=tests1[j][0].columns[:-2],
                                                  Cols_Chosen_For_Observation_Target=tests1[j][1].columns[:-2],
                                                  Poisson=Poisson)

                        perfs = Performance(scenario, tests1[j], alpha=alpha, algo=algo,
                                            type_supervision=type_supervision, Objective_Variable=Objective_Variable,
                                            Balance=Balance, Labelled_Proportion_Target=Labelled_Proportion_Target,
                                            Labelled_Proportion_Source=Labelled_Proportion_Source)
                        j = j + 1
                        z_coot = np.append(z_coot, perfs[0])
                        z_jdcoot = np.append(z_jdcoot, perfs[1])
                        test_z_coot = np.append(test_z_coot, perfs[2])
                        test_z_jdcoot = np.append(test_z_jdcoot, perfs[3])

                        if samp_s == R2[0]:
                            coot_perf = np.append(coot_perf, perfs[0])
                            jdcoot_perf = np.append(jdcoot_perf, perfs[1])
                            ref = np.append(ref, perfs[5])

                            test_coot_perf = np.append(test_coot_perf, perfs[2])
                            test_jdcoot_perf = np.append(test_jdcoot_perf, perfs[3])
                            test_ref = np.append(test_ref, perfs[4])

                MC_coot_perf = MC_coot_perf + coot_perf
                MC_jdcoot_perf = MC_jdcoot_perf + jdcoot_perf
                MC_ref_perf = MC_ref_perf + ref
                bp_coot = np.concatenate((bp_coot, np.array([coot_perf])), axis=1)
                bp_jdcoot = np.concatenate((bp_jdcoot, np.array([jdcoot_perf])), axis=1)

                test_MC_coot_perf = test_MC_coot_perf + test_coot_perf
                test_MC_jdcoot_perf = test_MC_jdcoot_perf + test_jdcoot_perf
                test_MC_ref_perf = test_MC_ref_perf + test_ref
                test_bp_coot = np.concatenate((test_bp_coot, np.array([test_coot_perf])), axis=1)
                test_bp_jdcoot = np.concatenate((test_bp_jdcoot, np.array([test_jdcoot_perf])), axis=1)

                hm_coot = hm_coot + z_coot.reshape((len(Sample_sizes), len(Sample_sizes)))
                hm_jdcoot = hm_jdcoot + z_jdcoot.reshape((len(Sample_sizes), len(Sample_sizes)))

                test_hm_coot = test_hm_coot + test_z_coot.reshape((len(Sample_sizes), len(Sample_sizes)))
                test_hm_jdcoot = test_hm_jdcoot + test_z_jdcoot.reshape((len(Sample_sizes), len(Sample_sizes)))

            MC_ref_perf = MC_ref_perf / Monte_Carlo
            MC_coot_perf = MC_coot_perf / Monte_Carlo
            MC_jdcoot_perf = MC_jdcoot_perf / Monte_Carlo

            test_MC_ref_perf = test_MC_ref_perf / Monte_Carlo
            test_MC_coot_perf = test_MC_coot_perf / Monte_Carlo
            test_MC_jdcoot_perf = test_MC_jdcoot_perf / Monte_Carlo

            hm_coot = hm_coot / Monte_Carlo
            hm_jdcoot = hm_jdcoot / Monte_Carlo
            test_hm_coot = test_hm_coot / Monte_Carlo
            test_hm_jdcoot = test_hm_jdcoot / Monte_Carlo

            Data = (
            MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, hm_coot, hm_jdcoot, test_MC_coot_perf,
            test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot, test_hm_coot, test_hm_jdcoot)

        graph_multi(Data, algo, Sample_sizes, Objective_Variable, "Mean shift variation impact",
                    "$i \ s.t. \mu_T=\mu_S+i$ \n with $\mu_S$ = " + str(R2[0]), "mean shift")

        if Store_in:
            Store(Data, "Mean_Shift_Variation")

        return (Data)

    else:

        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        if isinstance(Data, str):
            Data = Read(Data)
            Store_in = False
        if (type(Data) is tuple and len(Data) != 2):
            if len(Data) == 10:
                MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot = Data
                Store_in = False
            elif len(Data) == 14:
                MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, hm_coot, hm_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot, test_hm_coot, test_hm_jdcoot = Data
                Store_in = False
            else:
                print("Wrong data format")
                return 0

        else:

            MC_coot_perf = np.repeat(0, len(Sample_sizes))
            MC_jdcoot_perf = np.repeat(0, len(Sample_sizes))
            bp_coot = np.array([[]])
            bp_jdcoot = np.array([[]])
            MC_ref_perf = np.repeat(0, len(Sample_sizes))
            test_MC_coot_perf = np.repeat(0, len(Sample_sizes))
            test_MC_jdcoot_perf = np.repeat(0, len(Sample_sizes))
            test_bp_coot = np.array([[]])
            test_bp_jdcoot = np.array([[]])
            test_MC_ref_perf = np.repeat(0, len(Sample_sizes))

            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                coot_perf = np.array([])
                jdcoot_perf = np.array([])
                ref = np.array([])

                test_coot_perf = np.array([])
                test_jdcoot_perf = np.array([])
                test_ref = np.array([])
                j = 0
                for samp_s in Sample_sizes:
                    scenario = data_generator(n_Source, n_Target, d_Source, d_Target, mean_X_Source,
                                              mean_X_Target + samp_s, mean_Y_Source,
                                              mean_Y_Target, rho_source_generation, rho_source_non_generation,
                                              rho_target_generation, rho_target_non_generation, Sparse_Rate,
                                              Odds_Ratio_Source=samp_s, Odds_Ratio_Target=samp_s, R2_Source=R2_Source,
                                              R2_Target=R2_Target,
                                              Observed_Covariates_Proportion_Source=Observed_Covariates_Proportion_Source,
                                              Observed_Covariates_Proportion_Target=Observed_Covariates_Proportion_Target,
                                              Indexes_Chosen_For_Generation=Indexes_Chosen_For_Generation,
                                              Cols_Chosen_For_Observation_Source=tests2[j][0].columns[:-2],
                                              Cols_Chosen_For_Observation_Target=tests2[j][1].columns[:-2],
                                              Poisson=Poisson)

                    perfs = Performance(scenario, tests2[j], alpha=alpha, algo=algo, type_supervision=type_supervision,
                                        Objective_Variable=Objective_Variable, Balance=Balance,
                                        Labelled_Proportion_Target=Labelled_Proportion_Target,
                                        Labelled_Proportion_Source=Labelled_Proportion_Source)
                    j = j + 1
                    coot_perf = np.append(coot_perf, perfs[0])
                    jdcoot_perf = np.append(jdcoot_perf, perfs[1])
                    ref = np.append(ref, perfs[5])

                    test_coot_perf = np.append(test_coot_perf, perfs[2])
                    test_jdcoot_perf = np.append(test_jdcoot_perf, perfs[3])
                    test_ref = np.append(test_ref, perfs[4])

                MC_coot_perf = MC_coot_perf + coot_perf
                MC_jdcoot_perf = MC_jdcoot_perf + jdcoot_perf
                MC_ref_perf = MC_ref_perf + ref
                bp_coot = np.concatenate((bp_coot, np.array([coot_perf])), axis=1)
                bp_jdcoot = np.concatenate((bp_jdcoot, np.array([jdcoot_perf])), axis=1)

                test_MC_coot_perf = test_MC_coot_perf + test_coot_perf
                test_MC_jdcoot_perf = test_MC_jdcoot_perf + test_jdcoot_perf
                test_MC_ref_perf = test_MC_ref_perf + test_ref
                test_bp_coot = np.concatenate((test_bp_coot, np.array([test_coot_perf])), axis=1)
                test_bp_jdcoot = np.concatenate((test_bp_jdcoot, np.array([test_jdcoot_perf])), axis=1)

            MC_ref_perf = MC_ref_perf / Monte_Carlo
            MC_coot_perf = MC_coot_perf / Monte_Carlo
            MC_jdcoot_perf = MC_jdcoot_perf / Monte_Carlo

            test_MC_ref_perf = test_MC_ref_perf / Monte_Carlo
            test_MC_coot_perf = test_MC_coot_perf / Monte_Carlo
            test_MC_jdcoot_perf = test_MC_jdcoot_perf / Monte_Carlo

        if algo == 'COOT':
            tit = "Test Accuracy vs \n mean shift variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2)) + 1, test_MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Pure Accuracy vs \n mean shift variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2)) + 1, MC_coot_perf, color="red", label="COOT Mean Performance")

            axs[0].boxplot([bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="red"))

        if algo == 'JDCOOT':
            tit = "Test Accuracy vs v mean shift variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2)) + 1, test_MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="green"))

            tit = "Pure Accuracy vs \n mean shift variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with \mu_S = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2)) + 1, MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")

            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="green"))

        if algo == 'both':
            tit = "Test Accuracy vs v mean shift variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2)) + 1, test_MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="red"))

            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2)) + 1, MC_coot_perf, color="red", label="COOT Mean Performance")

            axs[0].boxplot([bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="red"))

            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[1].plot(np.arange(len(R2)) + 1, test_MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="green"))

            tit = "Pure Accuracy vs \n mean shift variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$i \ s.t. \mu_T=\mu_S+i$ with $\mu_S$ = " + str(R2[0]))
            axs[0].plot(np.arange(len(R2)) + 1, MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")

            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(R2))], patch_artist=True,
                           boxprops=dict(facecolor="green"))

        axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_ref_perf, '--', color="blue", label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_ref_perf, '--', color="blue", label="Reference")

        axs[0].legend(loc="best", fontsize="5")
        axs[0].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        # fig.show()
        if Store_in:
            Store((
                  MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf,
                  test_MC_ref_perf, test_bp_coot, test_bp_jdcoot), "Mean_Shift_Variation")
        return (MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)
