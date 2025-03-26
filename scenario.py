import math
import numpy as np
import pandas as pd
from jdcoot.data_scenario import DataScenario, DataScenarioTest

def generate_data(seed = 2025):

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
        test_scenario = DataScenarioTest()
    
        source, target = reference_scenario.generate(INDEX_GENERATION)
        source.to_csv("source.csv", index = False)
        target.to_csv("target.csv", index = False)
    
        source_test, target_test = test_scenario.generate(INDEX_GENERATION)
        source_test.to_csv("source_test.csv", index = False)
        target_test.to_csv("target_test.csv", index = False)
    
    
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

    return source, target, source_test, target_test


if __name__ == '__main__':

    source, target, test_source, test_target  = generate_data()

    print(source.columns)
    print(test_source.columns)
    print(target.columns)
    print(test_target.columns)

