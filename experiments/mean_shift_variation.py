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

np.random.seed(1972)

indices = np.random.choice(np.arange(100), math.ceil(0.75 * 100), replace=False)

nsimulations = 100

train_size = 1000
test_size = 1000 // 4

reference_scenario = DataScenario()
reference_scenario.size_source = train_size
reference_scenario.size_target = train_size
test_scenario = DataScenarioTest()
test_scenario.size_source = test_size
test_scenario.size_target = test_size
    
json_file = 'mean_shift_variation.json'
with open(json_file, 'w+') as f:
    f.seek(0)

mean_values = [0, 0.1, 0.2, 0.3, 0.4]

for i in range(nsimulations):
    
    for mean_shift_source in mean_values:
        for mean_shift_target in mean_values:

            reference_scenario.mean_x_source += mean_shift_source
            reference_scenario.mean_x_target += mean_shift_target

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
                            results = reference_scenario.__dict__.deepcopy()
                            results.pop('mean_x_source', None)
                            results.pop('mean_x_target', None)
                            results['variable'] = variable_type 
                            results['recoding'] = recoding_method
                            results['learning'] = learning_method 
                            results['prop_source'] = prop_source
                            results['prop_target'] = prop_target

                            pure, test = otrecod( source, target, source_test, target_test, 
                                               prop_source = prop_source, prop_target = prop_target)
                            results['train_size'] = train_size
                            results['test_sise'] = test_size
                            results['pure'] = pure
                            results['test'] = test
                            with open(json_file, 'a') as f:
                                json.dump(results, f)
                                f.write("\n")
                            
                        except KeyError:
                            print(f"Model : {variable_type}_{learning_method}_{recoding_method} is not available")
                            pass
    

with open(json_file) as f:

    lines = [json.loads(line) for line in f]
        
    data = pd.DataFrame(lines)

    data = data.rename( columns = {"pure":"accuracy"})

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data = data,
            x = "prop_source", y = "accuracy", hue = "recoding", showfliers = False)
    plt.savefig("observed_labels_proportions_variation.png")
