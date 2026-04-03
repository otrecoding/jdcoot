import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))

import jdcoot

data = jdcoot.generate_data()

prop_source=0.5
prop_target=0.005 

source, target, _, _ = data
n_target = len(target.Z)
n_source = len(source.Z)
l_source_train, l_source_test = train_test_split( np.arange(n_source), train_size=prop_source)
l_target_train, l_target_test = train_test_split( np.arange(n_target), train_size=prop_target)

pure_source, pure_target, test_source, test_target = jdcoot.continuous_partial_coot(
    *data, l_source_train, l_source_test, l_target_train, l_target_test, algo = "emd")

 

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
