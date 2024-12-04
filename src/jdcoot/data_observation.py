import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

from .performance import Performance
from .read_data import Read
from .store_data import Store

# -

# # Experiments and Graphics 

# ### Data observation variation

# +


def Semi_Supervised_Labelled_Proportion_Variations(Data, Data_test=None,
                                                   Proportion_Labelled_Variations=np.array([0, 0.5, 0.9]),
                                                   Monte_Carlo=1, algo='both', Objective_Variable='discrete',
                                                   Balance=False, alpha=None):
    """
    Semi_Supervised_Labelled_Proportion_Variations : Compute and plot performance versus the proportion of labelled observations in target

    Input : Data (tuple or str) : Learning data (tuple of 2), tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

                                OR

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

            Data_test (tuple) : Test data, tuple of 2 dataframes (Source in index 1 and Target in index 2) with the following format :

                                            | X_1 | ... | X_d | Y | Z |

                                        with X_i the ith observed covariate,
                                             Y the continuous objective variable for regression analysis (optional),
                                             Z the discrete objective variable for classification analysis (optional)

                                If None, Learning data is splited into 30% of test data and 70% of learning data


            Proportion_Labelled_Variations : np array of values of labelled proportions to vary

            Monte_Carlo : Number of repetitions of computing

            Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation
                                                                    if 'continuous', measure the performance for regression

            algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT
                                                        if 'JDCOOT', measure the performance for JDCOOT
                                                        if 'both',  measure the performance for COOT and JDCOOT

            Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data.

            alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())




    Output : Data (tuple) : tuple of 10  with respectively : Pure performance Monte Carlo COOT,
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

    nPoints = len(Proportion_Labelled_Variations)
    proportions = Proportion_Labelled_Variations
    Store_in = True

    if Objective_Variable == 'both':
        return -1

    if Monte_Carlo >= 1:

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

            MC_coot_perf = np.repeat(0, nPoints)
            MC_jdcoot_perf = np.repeat(0, nPoints)
            bp_coot = np.array([[]])
            bp_jdcoot = np.array([[]])
            MC_ref_perf = np.repeat(0, nPoints)

            test_MC_coot_perf = np.repeat(0, nPoints)
            test_MC_jdcoot_perf = np.repeat(0, nPoints)
            test_bp_coot = np.array([[]])
            test_bp_jdcoot = np.array([[]])
            test_MC_ref_perf = np.repeat(0, nPoints)

            with alive_bar(Monte_Carlo * nPoints, bar="classic2", spinner="twirls") as bar:
                for i in range(Monte_Carlo):
                    print("Entering Monte Carlo loop number {}".format(i))
                    coot_perf = np.array([])
                    jdcoot_perf = np.array([])
                    ref_perf = np.array([])
                    test_coot_perf = np.array([])
                    test_jdcoot_perf = np.array([])
                    test_ref_perf = np.array([])

                    for prop in proportions:
                        perf = Performance(Data, Data_test=Data_test, alpha=alpha, algo=algo,
                                           type_supervision='semi-supervised', Objective_Variable=Objective_Variable,
                                           Balance=Balance, Labelled_Proportion_Target=prop)
                        coot_perf = np.append(coot_perf, perf[0])
                        jdcoot_perf = np.append(jdcoot_perf, perf[1])
                        ref_perf = np.append(ref_perf, perf[5])

                        test_coot_perf = np.append(test_coot_perf, perf[2])
                        test_jdcoot_perf = np.append(test_jdcoot_perf, perf[3])
                        test_ref_perf = np.append(test_ref_perf, perf[4])

                        bar()
                    MC_coot_perf = MC_coot_perf + coot_perf
                    MC_jdcoot_perf = MC_jdcoot_perf + jdcoot_perf
                    MC_ref_perf = MC_ref_perf + ref_perf
                    bp_coot = np.concatenate((bp_coot, np.array([coot_perf])), axis=1)
                    bp_jdcoot = np.concatenate((bp_jdcoot, np.array([jdcoot_perf])), axis=1)

                    test_MC_coot_perf = test_MC_coot_perf + test_coot_perf
                    test_MC_jdcoot_perf = test_MC_jdcoot_perf + test_jdcoot_perf
                    test_MC_ref_perf = test_MC_ref_perf + test_ref_perf
                    test_bp_coot = np.concatenate((test_bp_coot, np.array([test_coot_perf])), axis=1)
                    test_bp_jdcoot = np.concatenate((test_bp_jdcoot, np.array([test_jdcoot_perf])), axis=1)

                MC_ref_perf = MC_ref_perf / Monte_Carlo
                MC_coot_perf = MC_coot_perf / Monte_Carlo
                MC_jdcoot_perf = MC_jdcoot_perf / Monte_Carlo

                test_MC_ref_perf = test_MC_ref_perf / Monte_Carlo
                test_MC_coot_perf = test_MC_coot_perf / Monte_Carlo
                test_MC_jdcoot_perf = test_MC_jdcoot_perf / Monte_Carlo

        if Objective_Variable == "discrete":
            tit2 = "Accuracy"
        else:
            tit2 = "MSE"

        if algo == 'both' or algo == 'COOT':
            tit = "Test Accuracy of methods versus the proportion of target data labelled"
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].plot(np.arange(nPoints) + 1, test_MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(nPoints)], patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Pure Accuracy of methods versus the proportion of target data observed"
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[1].set_xlabel("Proportion of target observation labelled")
            axs[0].plot(np.arange(nPoints) + 1, MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(nPoints)], patch_artist=True,
                           boxprops=dict(facecolor="red"))

        if algo == 'both' or algo == 'JDCOOT':
            tit = "Test Accuracy of methods versus the proportion of target data labelled"
            axs[1].set_title(tit)
            axs[1].set_ylabel(tit2)
            axs[1].plot(np.arange(nPoints) + 1, test_MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(nPoints)], patch_artist=True,
                           boxprops=dict(facecolor="green"))

            tit = "Pure Accuracy of methods versus the proportion of target data labelled"
            axs[0].set_title(tit)
            axs[0].set_ylabel(tit2)
            axs[1].set_xlabel("Proportion of target observation labelled")
            axs[0].plot(np.arange(nPoints) + 1, MC_jdcoot_perf, color="green", label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(nPoints)], patch_artist=True,
                           boxprops=dict(facecolor="green"))

        axs[0].plot(np.arange(nPoints) + 1, MC_ref_perf, '--', color="blue", label="Reference")
        axs[1].plot(np.arange(nPoints) + 1, test_MC_ref_perf, '--', color="blue", label="Reference")

        axs[1].legend(loc="best", fontsize="5")
        axs[0].legend(loc="best", fontsize="5")

        axs[0].set_xticks(np.arange(nPoints) + 1, labels=np.around(proportions, 2))
        axs[1].set_xticks(np.arange(nPoints) + 1, labels=np.around(proportions, 2))

        plt.tight_layout()
        plt.show()
        if Store_in:
            Store((
                  MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf,
                  test_MC_ref_perf, test_bp_coot, test_bp_jdcoot), "Semi_Supervised_Labelled_Proportion_Variations")
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
