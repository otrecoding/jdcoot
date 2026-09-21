import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

import jdcoot

prop_source = 1.0
prop_target = 0.1

source, target, source_test, target_test = jdcoot.generate_data()
n_target = len(target.Z)
n_source = len(source.Z)

l_source_train, l_source_test = None, None

l_target_train, l_target_test = train_test_split(
    np.arange(n_target), train_size=prop_target
)

pure_source, pure_target, test_source, test_target = jdcoot.discrete_semisupervised_coot(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
