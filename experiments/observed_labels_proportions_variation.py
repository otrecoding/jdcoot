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

for i in range(nsimulations):
    
    train_size = 1000
    test_size = 1000 // 5
    
    reference_scenario = DataScenario()
    reference_scenario.size_source = train_size
    reference_scenario.size_target = train_size
    test_scenario = DataScenarioTest()
    test_scenario.size_source = test_size
    test_scenario.size_target = test_size
    
    source, target = reference_scenario.generate(indices)
    
    source_test, target_test = test_scenario.generate(indices)
        
    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]
    
    proportions = [0.1, 0.5, 0.9]
    
    json_file = 'observed_labels_proportions_variation.json'
    with open(json_file, 'w+') as f:
        f.seek(0)

    for prop_source in proportions:
        for prop_target in proportions:
            for variable_type in variable_types:
                for learning_method in learning_methods:
                    for recoding_method in recoding_methods:
                        try:
                            print(f"Model : {variable_type}_{learning_method}_{recoding_method}")
                            otrecod = models[(variable_type, learning_method, recoding_method)]
                            results = dict(variable = variable_type, 
                                 recoding = recoding_method,
                                 learning = learning_method, 
                                 prop_source = prop_source,
                                 prop_target = prop_target)

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
