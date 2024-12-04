import os

import numpy as np


def Store(Data, name):
    """
    Store(Data,name) : Store Data from a scenario
    
    Input : Data (tuple) : tuple of 10 or 14 elements with respectively : For Multivariation Scenarios (Heatmap) : 14 elements :  
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
                                                                            MC_coot_perf,
                                                                            MC_jdcoot_perf,
                                                                            MC_ref_perf,
                                                                            bp_coot,
                                                                            bp_jdcoot,
                                                                            test_MC_coot_perf,
                                                                            test_MC_jdcoot_perf,
                                                                            test_MC_ref_perf,
                                                                            test_bp_coot,
                                                                            test_bp_jdcoot
            name (str) : Name of the simulation to give in the folder (only the name, not the path)
    """

    if not os.path.isdir(os.getcwd() + "\\Stored_Data\\"):
        os.mkdir(os.getcwd() + "\\Stored_Data\\")

    if os.path.isdir(os.getcwd() + "\\Stored_Data\\" + name + "\\"):
        for i in range(1, 1000):
            if not os.path.isdir(os.getcwd() + "\\Stored_Data\\" + name + "_" + str(i) + "\\"):
                name = name + "_" + str(i)
                print("Existing save, creating " + name)
                break

    os.mkdir(os.getcwd() + "\\Stored_Data\\" + name + "\\")
    for i in range(len(Data)):
        np.save(os.getcwd() + "\\Stored_Data\\" + name + "\\" + str(i), Data[i], allow_pickle=True, fix_imports=True)
    print("Saved as : " + os.getcwd() + "\\Stored_Data\\" + name)
