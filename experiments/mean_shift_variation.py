import math
import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
sys.path.append(os.path.abspath('src'))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import compute

nsimulations = 200

train_scenario = DataScenario()
test_scenario = DataScenarioTest()
    
json_file = 'mean_shift_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot"]

mean_source_values = [0.0]
mean_target_values = [0.0, 0.1, 0.2, 0.3, 0.4]

for i in range(nsimulations):

    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)
    
    for mean_shift_source in mean_source_values:
        for mean_shift_target in mean_target_values:

            train_scenario.mean_x_source.fill(0.0)
            train_scenario.mean_x_target.fill(0.0)

            train_scenario.mean_x_source += mean_shift_source
            train_scenario.mean_x_target += mean_shift_target

            test_scenario.mean_x_source.fill(0.0)
            test_scenario.mean_x_target.fill(0.0)

            test_scenario.mean_x_source += mean_shift_source
            test_scenario.mean_x_target += mean_shift_target

            compute(json_file, train_scenario, test_scenario, indices, 
                    variable_types, learning_methods, recoding_methods)
    

