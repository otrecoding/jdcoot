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
from copy import deepcopy

np.random.seed(1972)

nsimulations = 1

prop_source, prop_target = 0.2, 0.2
sizes = [10, 100, 500, 1000] 
test_size = 300
reference_scenario = DataScenario()
test_scenario = DataScenarioTest()

json_file = 'sample_size_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

for i in range(nsimulations):
    
    indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

    for size_source in sizes:
        for size_target in sizes:

            reference_scenario.size_source = size_source
            reference_scenario.size_target = size_target
            test_scenario.size_source = test_size
            test_scenario.size_target = test_size
            
            source, target = reference_scenario.generate(indices)
            source_test, target_test = test_scenario.generate(indices)
            source_test = source_test.loc[:, source.columns]
            target_test = target_test.loc[:, target.columns]
            
            for variable_type in variable_types:
                for learning_method in learning_methods:
                    for recoding_method in recoding_methods:
                        try:
                            otrecod = models[(variable_type, learning_method, recoding_method)]
                            print(f"Model : {variable_type}_{learning_method}_{recoding_method}")
                            results = deepcopy(reference_scenario.__dict__)
                            results.pop('mean_x_source', None)
                            results.pop('mean_x_target', None)
                            results['variable'] = variable_type 
                            results['recoding'] = recoding_method
                            results['learning'] = learning_method 
                            results['prop_source'] = prop_source
                            results['prop_target'] = prop_target

                            pure, test = otrecod( source, target, source_test, target_test, 
                                                  prop_source = prop_source, prop_target = prop_target)

                            results['pure'] = pure
                            results['test'] = test
                            results['train_size_source'] = size_source
                            results['train_size_target'] = size_target
                            results['test_size'] = test_size
                            with open(json_file, 'a') as f:
                                json.dump(results, f)
                                f.write("\n")
                            
                        except KeyError:
                            print(f"Model : {variable_type}_{learning_method}_{recoding_method} is not available")
                            pass
