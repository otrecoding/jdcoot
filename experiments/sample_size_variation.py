import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))
from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

np.random.seed(1972)

nsimulations = 1

prop_source, prop_target = 0.2, 0.2
sizes = [10, 100, 500, 1000] 
train_scenario = DataScenario()
test_scenario = DataScenarioTest()

json_file = 'sample_size_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

for i in range(nsimulations):
    
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    for size_source in sizes:
        for size_target in sizes:

            train_scenario.size_source = size_source
            train_scenario.size_target = size_target
            test_scenario.size_source = size_source // 3
            test_scenario.size_target = size_target // 3
            
            compute(json_file, train_scenario, test_scenario, indices)
