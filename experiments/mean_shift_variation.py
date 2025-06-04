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
from jdcoot.performance import models, variable_types, learning_methods, recoding_methods


indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

nsimulations = 100

train_scenario = DataScenario()
test_scenario = DataScenarioTest()
    
json_file = 'mean_shift_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

mean_values = [0, 0.1, 0.2, 0.3, 0.4]

for i in range(nsimulations):
    
    for mean_shift_source in mean_values:

        for mean_shift_target in mean_values:

            train_scenario.mean_x_source += mean_shift_source
            train_scenario.mean_x_target += mean_shift_target

            test_scenario.mean_x_source += mean_shift_source
            test_scenario.mean_x_target += mean_shift_target

            compute(json_file, train_scenario, test_scenario, indices )
    

