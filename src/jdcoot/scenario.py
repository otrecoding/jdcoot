import math
import numpy as np
import pandas as pd
from .data_scenario import DataScenario, DataScenarioTest

def generate_data(size = 1000, seed = 2025):

    np.random.seed(seed)

    INDEX_GENERATION = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    try:
        source = pd.read_csv("source.csv")
        target = pd.read_csv("target.csv")
    
        source_test = pd.read_csv("source_test.csv")
        source_test = source_test.loc[:, source.columns]
        target_test = pd.read_csv("target_test.csv")
        target_test = target_test.loc[:, target.columns]
    
    except FileNotFoundError:
    
        reference_scenario = DataScenario()
        reference_scenario.size_source = size
        reference_scenario.size_target = size
        test_scenario = DataScenarioTest()
        test_scenario.size_source = size // 5
        test_scenario.size_target = size // 5
    
        source, target = reference_scenario.generate(INDEX_GENERATION)
        source.to_csv("source.csv", index = False)
        target.to_csv("target.csv", index = False)
    
        source_test, target_test = test_scenario.generate(INDEX_GENERATION)
        source_test.to_csv("source_test.csv", index = False)
        target_test.to_csv("target_test.csv", index = False)
    
    
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

    return source, target, source_test, target_test


def select_observations(source, target):

        z_source = source.Z.values
        z_target = target.Z.values

        n_source = len(z_source)
        n_target = len(z_target)

        z_levels = np.sort(np.union1d(np.unique(z_source), np.unique(z_target)))
        print(f' levels = {z_levels}')

        if len(z_levels) > 2:

            del_idx = []
            for k in z_levels:
                if sum(z_source == k) < 0.01 * n_source:
                    del_idx.append(k)
            source = source.loc[~np.in1d(z_source, del_idx), :].reset_index(drop=True)

        if len(z_levels) > 2:
            del_idx = []
            for k in z_levels:
                if sum(z_target == k) < 0.01 * n_target:
                    del_idx.append(k)
            target = target.loc[~np.isin(z_target, del_idx), :].reset_index(drop=True)

        S_nPerClass = math.ceil(min(np.unique(z_source, return_counts=True)[1]))  
        T_nPerClass = math.ceil(min(np.unique(z_target, return_counts=True)[1]))

        z_kept_source = np.array([]).astype(int)
        for lab in np.unique(z_source):
            z_kept_source = np.append(z_kept_source,
                                      np.random.choice(np.where(z_source == lab)[0], S_nPerClass, replace=False))

        source = source.loc[z_kept_source, :].reset_index(drop=True)

        z_kept_target = np.array([]).astype(int)
        for lab in z_levels:
            z_kept_target = np.append(z_kept_target, np.random.choice(np.where(z_target == lab)[0], T_nPerClass, replace=False))
        target = target.loc[z_kept_target, :].reset_index(drop=True)

        return source, target



if __name__ == '__main__':

    source, target, test_source, test_target  = generate_data()

    print(source.columns)
    print(test_source.columns)
    print(target.columns)
    print(test_target.columns)

