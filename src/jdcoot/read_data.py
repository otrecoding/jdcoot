import os

import numpy as np


def Read(name):
    """
    Read(name) : Read Data from an existing save
    
    Input : name (str) : Name of the simulation given in the folder (only the name, not the path)
    
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
    if not os.path.isdir(os.getcwd() + "\\Stored_Data\\" + name + "\\"):
        print("Not existing save")
        return 0
    else:
        Data = []
        for i in range(len(os.listdir(os.getcwd() + "\\Stored_Data\\" + name + "\\"))):
            Data.append(np.load(os.getcwd() + "\\Stored_Data\\" + name + "\\" + str(i) + ".npy"))
    return tuple(Data)
