from copy import deepcopy
import json
import numpy as np

from .models.discrete_partial_coot import discrete_partial_coot
from .models.discrete_partial_jdcoot import discrete_partial_jdcoot
from .models.discrete_partial_reference import discrete_partial_reference
from .models.discrete_semisupervised_coot import discrete_semisupervised_coot
from .models.discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from .models.discrete_semisupervised_reference import discrete_semisupervised_reference
from .models.discrete_unsupervised_coot import discrete_unsupervised_coot
from .models.discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot

models = dict()
models[("discrete", "unsupervised", "coot")] = discrete_unsupervised_coot
models[("discrete", "unsupervised", "jdcoot")] = discrete_unsupervised_jdcoot
models[("discrete", "semisupervised", "coot")] = discrete_semisupervised_coot
models[("discrete", "semisupervised", "jdcoot")] = discrete_semisupervised_jdcoot
models[("discrete", "semisupervised", "reference")] = discrete_semisupervised_reference
models[("discrete", "partial", "coot")] = discrete_partial_coot
models[("discrete", "partial", "jdcoot")] = discrete_partial_jdcoot
models[("discrete", "partial", "reference")] = discrete_partial_reference


def compute(
    json_file,
    train,
    test,
    indices,
    learning_methods,
    recoding_methods,
    **kwargs,
):
    prop_source = kwargs.get("prop_source", np.nan)
    prop_target = kwargs.get("prop_target", np.nan)

    source, target = train.generate(indices)
    source_test, target_test = test.generate(indices)

    source_test = source_test.loc[:, source.columns]
    target_test = target_test.loc[:, target.columns]

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
                results["prop_source"] = prop_source
                results["prop_target"] = prop_target
                results["sparse_rate"] = len(indices) / 100

                pure_source, pure_target, test_source, test_target = otrecod(
                    source,
                    target,
                    source_test,
                    target_test,
                    prop_source=prop_source,
                    prop_target=prop_target,
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

    return True
