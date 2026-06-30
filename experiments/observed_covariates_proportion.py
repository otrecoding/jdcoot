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

train = DataScenario()
test = DataScenarioTest()

json_file = "observed_covariates_proportion.json"
with open(json_file, "w+") as f:
    f.seek(0)

variable_types = ["continuous", "discrete"]
learning_methods = ["unsupervised"]
recoding_methods = ["coot", "jdcoot", "reference"]
sparse_rate = 0.75
prop_source, prop_target = 1.0, 0.0

values = [0.2, 0.4, 0.6, 0.8]

for i in range(nsimulations):
    indices = np.random.choice(
        np.arange(100), math.ceil(sparse_rate * 100), replace=False
    )

    for pxo_source in values:
        for pxo_target in values:
            train.obs_covar_prop_source = pxo_source
            train.obs_covar_prop_target = pxo_target
            source, target = train.generate(indices)
            source_test, target_test = test.generate(indices)

            source_test = source_test.loc[:, source.columns]
            target_test = target_test.loc[:, target.columns]

            for variable_type in variable_types:
                for learning_method in learning_methods:
                    for recoding_method in recoding_methods:
                        try:
                            otrecod = models[
                                (variable_type, learning_method, recoding_method)
                            ]
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
                            results["prop_source"] = prop_source
                            results["prop_target"] = prop_target

                            pure_source, pure_target, test_source, test_target = (
                                otrecod(
                                    source,
                                    target,
                                    source_test,
                                    target_test,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
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
