import matplotlib.pyplot as plt
import numpy as np


def graph_multi(Data, algo, TAB, Objective_Variable, maintit, metric, name_metric):
    """
    graph_multi : to plot graph, called in other functions
    """
    Proportion_Labelled_Variations = TAB
    Sample_sizes = TAB

    MC_coot_perf, MC_jdcoot_perf, MC_ref_perf, bp_coot, bp_jdcoot, hm_coot, hm_jdcoot, test_MC_coot_perf, test_MC_jdcoot_perf, test_MC_ref_perf, test_bp_coot, test_bp_jdcoot, test_hm_coot, test_hm_jdcoot = Data

    if Objective_Variable == "discrete":
        tit2 = "Accuracy"
    else:
        tit2 = "MSE"

    if True:

        if algo == 'COOT':
            fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
            fig.suptitle(maintit, fontsize=15)
            tit = "Test " + tit2
            axs[1, 0].set_title(tit)
            axs[1, 0].set_ylabel(tit2)
            axs[1, 0].set_xlabel(metric)
            axs[1, 0].plot(np.arange(len(Proportion_Labelled_Variations)) + 1, test_MC_coot_perf, color="red",
                           label="COOT")
            axs[1, 0].boxplot([test_bp_coot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="red"))

            tit = "Pure " + tit2
            axs[0, 0].set_title(tit)
            axs[0, 0].set_ylabel(tit2)
            axs[0, 0].set_xlabel(metric)
            axs[0, 0].plot(np.arange(len(Sample_sizes)) + 1, MC_coot_perf, color="red", label="COOT")
            axs[0, 0].boxplot([bp_coot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="red"))

            min_ = min(np.min(hm_coot), np.min(test_hm_coot))
            max_ = max(np.max(hm_coot), np.max(test_hm_coot))

            tit = "HM for COOT (Pure)"
            axs[0, 1].set_title(tit)
            axs[0, 1].set_xlabel("Target data " + name_metric)
            axs[0, 1].set_ylabel("Source data " + name_metric)
            c1 = axs[0, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), hm_coot, vmin=min_,
                                      vmax=max_, cmap="viridis")
            axs[0, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[0, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0, 1])

            tit = "HM for COOT (test)"
            axs[1, 1].set_title(tit)
            axs[1, 1].set_xlabel("Target data " + name_metric)
            axs[1, 1].set_ylabel("Source data " + name_metric)
            axs[1, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), test_hm_coot, vmin=min_,
                                 vmax=max_, cmap="viridis")
            axs[1, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[1, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1, 1])

            fs = "6"

        if algo == 'JDCOOT':
            fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
            fig.suptitle(maintit, fontsize=15)

            tit = "Test " + tit2
            axs[1, 0].set_title(tit)
            axs[1, 0].set_ylabel(tit2)
            axs[1, 0].set_xlabel(metric)
            axs[1, 0].plot(np.arange(len(Sample_sizes)) + 1, test_MC_jdcoot_perf, color="green", label="JDCOOT")
            axs[1, 0].boxplot([test_bp_jdcoot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="green"))

            tit = "Pure " + tit2
            axs[0, 0].set_title(tit)
            axs[0, 0].set_ylabel(tit2)
            axs[0, 0].set_xlabel(metric)
            axs[0, 0].plot(np.arange(len(Sample_sizes)) + 1, MC_jdcoot_perf, color="green", label="JDCOOT")
            axs[0, 0].boxplot([bp_jdcoot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="green"))

            min_ = min(np.min(hm_jdcoot), np.min(test_hm_jdcoot))
            max_ = max(np.max(hm_jdcoot), np.max(test_hm_jdcoot))

            tit = "HM for JDCOOT (pure)"
            axs[0, 1].set_title(tit)
            axs[0, 1].set_xlabel("Target data " + name_metric)
            axs[0, 1].set_ylabel("Source data " + name_metric)
            c1 = axs[0, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), hm_jdcoot, vmin=min_,
                                      vmax=max_, cmap="viridis")
            axs[0, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[0, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0, 1])

            tit = "HM for JDCOOT (test)"
            axs[1, 1].set_title(tit)
            axs[1, 1].set_xlabel("Target data " + name_metric)
            axs[1, 1].set_ylabel("Source data " + name_metric)
            axs[1, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), test_hm_jdcoot, vmin=min_,
                                 vmax=max_, cmap="viridis")
            axs[1, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            axs[1, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1, 1])

            fs = "6"

        if algo == 'both':
            fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
            fig.suptitle(maintit, fontsize=15)

            axs[1, 0].set_ylabel(tit2)
            axs[1, 0].set_xlabel(metric)
            axs[1, 0].plot(np.arange(len(Sample_sizes)) + 1, test_MC_coot_perf, color="red", label="COOT")
            axs[1, 0].boxplot([test_bp_coot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="red"))

            axs[0, 0].set_ylabel(tit2)
            axs[0, 0].set_xlabel(metric)
            axs[0, 0].plot(np.arange(len(Sample_sizes)) + 1, MC_coot_perf, color="red", label="COOT")
            axs[0, 0].boxplot([bp_coot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="red"))

            tit = "Test " + tit2
            axs[1, 0].set_title(tit)
            axs[1, 0].set_ylabel(tit2)
            axs[1, 0].set_xlabel(metric)
            axs[1, 0].plot(np.arange(len(Sample_sizes)) + 1, test_MC_jdcoot_perf, color="green", label="JDCOOT")
            axs[1, 0].boxplot([test_bp_jdcoot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="green"))

            tit = "Pure " + tit2
            axs[0, 0].set_title(tit)
            axs[0, 0].set_ylabel(tit2)
            axs[0, 0].set_xlabel(metric)
            axs[0, 0].plot(np.arange(len(Sample_sizes)) + 1, MC_jdcoot_perf, color="green", label="JDCOOT")
            axs[0, 0].boxplot([bp_jdcoot.reshape((-1, len(Sample_sizes)))[:, i] for i in range(len(Sample_sizes))],
                              patch_artist=True,
                              boxprops=dict(facecolor="green"))

            min_ = min(np.min(hm_jdcoot), np.min(hm_coot))
            max_ = max(np.max(hm_jdcoot), np.max(hm_coot))
            min_test = min(np.min(test_hm_jdcoot), np.min(test_hm_coot))
            max_test = max(np.max(test_hm_jdcoot), np.max(test_hm_coot))

            Mi = min(min_, min_test)
            Ma = max(max_, max_test)

            tit = "HM for COOT (Pure)"
            axs[0, 1].set_title(tit)
            axs[0, 1].set_xlabel("Target data " + name_metric)
            axs[0, 1].set_ylabel("Source data " + name_metric)
            c1 = axs[0, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), hm_coot, vmin=Mi,
                                      vmax=Ma, cmap="viridis")
            axs[0, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str), rotation=45)
            axs[0, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0, 1])

            tit = "HM for JDCOOT (Pure)"
            axs[0, 2].set_title(tit)
            axs[0, 2].set_xlabel("Target data " + name_metric)
            axs[0, 2].set_ylabel("Source data " + name_metric)
            axs[0, 2].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), hm_jdcoot, vmin=Mi,
                                 vmax=Ma, cmap="viridis")

            axs[0, 2].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str), rotation=45)
            axs[0, 2].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[0, 2])

            tit = "HM for JDCOOT (Test)"
            axs[1, 2].set_title(tit)
            axs[1, 2].set_xlabel("Target data " + name_metric)
            axs[1, 2].set_ylabel("Source data " + name_metric)
            axs[1, 2].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), test_hm_jdcoot,
                                      vmin=Mi, vmax=Ma, cmap="viridis")
            axs[1, 2].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str), rotation=45)
            axs[1, 2].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1, 2])

            tit = "HM for COOT (Test)"
            axs[1, 1].set_title(tit)
            axs[1, 1].set_xlabel("Target data " + name_metric)
            axs[1, 1].set_ylabel("Source data " + name_metric)
            axs[1, 1].pcolormesh(np.arange(len(Sample_sizes)), np.arange(len(Sample_sizes)), test_hm_coot, vmin=Mi,
                                 vmax=Ma, cmap="viridis")
            axs[1, 1].set_xticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str), rotation=45)
            axs[1, 1].set_yticks(np.arange(len(Sample_sizes)), labels=Sample_sizes.astype(str))
            fig.colorbar(c1, ax=axs[1, 1])

            fs = "4"

        axs[0, 0].plot(np.arange(len(Sample_sizes)) + 1, MC_ref_perf, '--', color="blue", label="Reference")
        axs[1, 0].plot(np.arange(len(Sample_sizes)) + 1, test_MC_ref_perf, '--', color="blue", label="Reference")

        # axs[1,0].legend(loc="best",fontsize="7")
        axs[0, 0].legend(loc="best", fontsize=fs)
        axs[0, 0].sharex(axs[1, 0])
        axs[0, 0].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str), rotation=45)
        axs[1, 0].set_xticks(np.arange(len(Sample_sizes)) + 1, labels=Sample_sizes.astype(str), rotation=45)
        fig.tight_layout()
        # fig.show()
