from copy import deepcopy
import json
import math
import numpy as np
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

from jdcoot import DataScenario, DataScenarioTest
from jdcoot.performance import models
from sklearn.model_selection import train_test_split

nsimulations = 100

size = 1000
size_source_train = size
size_target_train = size
size_source_test = size
size_target_test = size

train = DataScenario()
test = DataScenarioTest()

train.size_source = size_source_train
train.size_target = size_target_train
test.size_source = size_source_test
test.size_target = size_target_test

json_file = "sparse_rate_variation.json"
with open(json_file, "w+") as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot"]

prop_source, prop_target = 0.1, 0.1

for sparse_rate in [0.25, 0.5, 0.75, 1]:

    indices = np.random.choice(np.arange(100), math.ceil(sparse_rate * 100), replace=False)

    for i in range(nsimulations):

        source, target = train.generate(indices)
        source_test, target_test = test.generate(indices)

        source_test = source_test.loc[:, source.columns]
        target_test = target_test.loc[:, target.columns]

        n_source = len(source.Y)
        n_target = len(target.Y)
        l_source_train, l_source_test = train_test_split(np.arange(n_source), train_size=prop_source)
        l_target_train, l_target_test = train_test_split(np.arange(n_target), train_size=prop_target)

        for variable_type in variable_types:
            for learning_method in learning_methods:
                for recoding_method in recoding_methods:
                    try:
                        otrecod = models[(variable_type, learning_method, recoding_method)]
                        print(
                        f"Model : {variable_type}_{learning_method}_{recoding_method}"
                        )
                        results = deepcopy(train.__dict__)
                        results["mean_x_source"] = np.mean(train.mean_x_source)
                        results["mean_x_target"] = np.mean(train.mean_x_target)
                        results["variable"] = variable_type
                        results["recoding"] = recoding_method
                        results["learning"] = learning_method
                        results["sparse_rate"] = sparse_rate

                        pure_source, pure_target, test_source, test_target = otrecod(
                            source,
                            target,
                            source_test,
                            target_test,
                            l_source_train, l_source_test,
                            l_target_train, l_target_test,
                        )

                        results["pure_source"] = pure_source
                        results["test_source"] = test_source
                        results["pure_target"] = pure_target
                        results["test_target"] = test_target
                        results["size_source_test"] = test.size_source
                        results["size_target_test"] = test.size_target
                  
                        with open(json_file, "a") as f:
                            json.dump(results, f)
                            f.write("\n")

                    except KeyError:
                        print(
                            f"Model : {variable_type}_{learning_method}_{recoding_method} is not available"
                    )
                        pass
