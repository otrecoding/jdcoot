import matplotlib.pyplot as plt
import numpy as np

from .graph_multi import graph_multi
from .performance import Performance
from .read_data import Read
from .store_data import Store

# +


def Partial_Labelled_Proportion_Variations(Data, Data_test=None,
                                           Proportion_Labelled_Variations=np.array([0.1, 0.5, 0.9]), Monte_Carlo=1,
                                           algo='both', Objective_Variable='discrete', Multi_Variations=True,
                                           Balance=False, alpha=None):
    """
    Partial_Labelled_Proportion_Variations : Compute and plot performance versus the proportion of labelled observations in target and source

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


            Multi_Variations (bool) : if true, compute performance along source and target simultaneous variation of proportions of labelled observations, and plot heatmaps


            Objective_Variable (str : 'discrete' or 'continuous') : if 'discrete', measure the performance for classifisation
                                                                    if 'continuous', measure the performance for regression

            algo (str : 'COOT' or 'JDCOOT' or 'both') : if 'COOT', measure the performance for COOT
                                                        if 'JDCOOT', measure the performance for JDCOOT
                                                        if 'both',  measure the performance for COOT and JDCOOT

            Balance (bool) : if true, balance the number of observation in each class. So, source and target dscrete variables have the same distribution. However, it destroys some of the data.

            alpha (float) : Hyper parameter of the problem formulation as in the paper. If None, optimized alpha is used (optimized for data given by the previous function data_generator())





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

    Sample_sizes = Proportion_Labelled_Variations  # pratique
    Store_in = True
    if Objective_Variable == 'both':
        return -1

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
            MC_ref_perf = np.repeat(0, len(Sample_sizes))

            test_MC_coot_perf = np.repeat(0, len(Sample_sizes))
            test_MC_jdcoot_perf = np.repeat(0, len(Sample_sizes))
            test_bp_coot = np.array([[]])
            test_bp_jdcoot = np.array([[]])
            test_MC_ref_perf = np.repeat(0, len(Sample_sizes))

            for i in range(Monte_Carlo):
                print("Entering Monte Carlo loop number {}".format(i))
                test_z_coot = np.array([])
                test_z_jdcoot = np.array([])

                z_coot = np.array([])
                z_jdcoot = np.array([])
                coot_perf = np.array([])
                jdcoot_perf = np.array([])
                ref_perf = np.array([])

                test_coot_perf = np.array([])
                test_jdcoot_perf = np.array([])
                test_ref_perf = np.array([])
                for samp_s in Proportion_Labelled_Variations:
                    for samp_t in Proportion_Labelled_Variations:

                        perfs = Performance(Data, Data_test, alpha=alpha, algo=algo, type_supervision='partial',
                                            Objective_Variable=Objective_Variable, Balance=Balance,
                                            Labelled_Proportion_Target=samp_t, Labelled_Proportion_Source=samp_s)
                        z_coot = np.append(z_coot, perfs[0])
                        z_jdcoot = np.append(z_jdcoot, perfs[1])
                        test_z_coot = np.append(test_z_coot, perfs[2])
                        test_z_jdcoot = np.append(test_z_jdcoot, perfs[3])

                        if samp_t == samp_s:
                            coot_perf = np.append(coot_perf, perfs[0])
                            jdcoot_perf = np.append(jdcoot_perf, perfs[1])
                            ref_perf = np.append(ref_perf, perfs[5])

                            test_coot_perf = np.append(test_coot_perf, perfs[2])
                            test_jdcoot_perf = np.append(test_jdcoot_perf, perfs[3])
                            test_ref_perf = np.append(test_ref_perf, perfs[4])

                MC_ref_perf = MC_ref_perf + ref_perf
                MC_coot_perf = MC_coot_perf + coot_perf
                MC_jdcoot_perf = MC_jdcoot_perf + jdcoot_perf
                bp_coot = np.concatenate((bp_coot, np.array([coot_perf])), axis=1)
                bp_jdcoot = np.concatenate((bp_jdcoot, np.array([jdcoot_perf])), axis=1)

                test_MC_ref_perf = test_MC_ref_perf + test_ref_perf
                test_MC_coot_perf = test_MC_coot_perf + test_coot_perf
                test_MC_jdcoot_perf = test_MC_jdcoot_perf + test_jdcoot_perf
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

        graph_multi(Data, algo, Sample_sizes, Objective_Variable, "Observed labels proportion variation impact",
                    "$p^T=p^S=$Proportion of \n observed labels", "Observed \n labels proportion")

        if Store_in:
            Store((MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, hm_coot, hm_jdcoot, test_MC_coot_perf,
                   test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot, test_hm_coot, test_hm_jdcoot),
                  "Partial_Labelled_Proportion_Variations")

        return (MC_coot_perf,
                MC_jdcoot_perf,
                MC_ref_perf,
                bp_coot,
                bp_jdcoot,
                hm_coot,
                hm_jdcoot,
                test_MC_coot_perf,
                test_MC_jdcoot_perf,
                test_MC_ref_perf,
                test_bp_coot,
                test_bp_jdcoot)

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
                for samp_s in Sample_sizes:
                    perfs = Performance(Data, Data_test, alpha=alpha, algo=algo, type_supervision='partial',
                                        Objective_Variable=Objective_Variable, Balance=Balance,
                                        Labelled_Proportion_Target=samp_s, Labelled_Proportion_Source=samp_s)
                    coot_perf = np.append(coot_perf, perfs[0])
                    jdcoot_perf = np.append(jdcoot_perf, perfs[1])
                    ref = np.append(ref, perfs[5])

                    test_coot_perf = np.append(test_coot_perf, perfs[2])
                    test_jdcoot_perf = np.append(test_jdcoot_perf, perfs[3])
                    test_ref = np.append(test_ref, perfs[4])

                MC_ref_perf = MC_ref_perf + ref
                MC_coot_perf = MC_coot_perf + coot_perf
                MC_jdcoot_perf = MC_jdcoot_perf + jdcoot_perf
                bp_coot = np.concatenate((bp_coot, np.array([coot_perf])), axis=1)
                bp_jdcoot = np.concatenate((bp_jdcoot, np.array([jdcoot_perf])), axis=1)

                test_MC_ref_perf = test_MC_ref_perf + test_ref
                test_MC_coot_perf = test_MC_coot_perf + test_coot_perf
                test_MC_jdcoot_perf = test_MC_jdcoot_perf + test_jdcoot_perf
                test_bp_coot = np.concatenate((test_bp_coot, np.array([test_coot_perf])), axis=1)
                test_bp_jdcoot = np.concatenate((test_bp_jdcoot, np.array([test_jdcoot_perf])), axis=1)

            MC_ref_perf = MC_ref_perf / Monte_Carlo
            MC_coot_perf = MC_coot_perf / Monte_Carlo
            MC_jdcoot_perf = MC_jdcoot_perf / Monte_Carlo

            test_MC_ref_perf = test_MC_ref_perf / Monte_Carlo
            test_MC_coot_perf = test_MC_coot_perf / Monte_Carlo
            test_MC_jdcoot_perf = test_MC_jdcoot_perf / Monte_Carlo

        if algo == 'COOT':
            tit = "Test Accuracy vs Observed labels \n proportion variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Pure Accuracy vs Observed labels \n proportion variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="red"))

        if algo == 'JDCOOT':
            tit = "Test Accuracy vs Observed labels \n proportion variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_jdcoot_perf, color="green",
                        label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="green"))

            tit = "Pure Accuracy vs Observed labels \n proportion variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_jdcoot_perf, color="green",
                        label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="green"))

        if algo == 'both':
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of observed labels")
            axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[1].boxplot([test_bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Accuracy vs Observed labels \n proportion variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_coot_perf, color="red", label="COOT Mean Performance")
            axs[0].boxplot([bp_coot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Test Accuracy vs Observed labels \n proportion variations"
            axs[1].set_title(tit)
            axs[1].set_ylabel("Accuracy")
            axs[1].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_jdcoot_perf, color="green",
                        label="JDCOOT Mean Performance")
            axs[1].boxplot([test_bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="red"))

            tit = "Pure Accuracy vs Observed labels \n proportion variations"
            axs[0].set_title(tit)
            axs[0].set_ylabel("Accuracy")
            axs[0].set_xlabel("$p^T=p^S=$Proportion of \n observed labels")
            axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_jdcoot_perf, color="green",
                        label="JDCOOT Mean Performance")
            axs[0].boxplot([bp_jdcoot.reshape((Monte_Carlo, -1))[:, i] for i in range(len(Sample_sizes))],
                           patch_artist=True,
                           boxprops=dict(facecolor="green"))

        axs[0].plot(np.arange(len(Sample_sizes)) + 1, MC_ref_perf, '--', color="blue", label="Reference")
        axs[1].plot(np.arange(len(Sample_sizes)) + 1, test_MC_ref_perf, '--', color="blue", label="Reference")

        axs[1].legend(loc="best", fontsize="5")

        axs[0].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str))
        axs[1].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str))
        fig.tight_layout()
        # fig.show()
        if Store_in:
            Store((
                  MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf,
                  test_MC_ref_perf, test_bp_coot, test_bp_jdcoot), "Partial_Labelled_Proportion_Variations")

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
