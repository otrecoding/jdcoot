import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

from jdcoot.scenario import generate_data

source, target, source_test, target_test = generate_data()

l_source_train, l_source_test = None, None

n_target = len(target.Z)
l_target_train, l_target_test = train_test_split(np.arange(n_target), train_size=0.01)
alpha = 0.1

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_semisupervised_jdcoot(
        source,
        target,
        source_test,
        target_test,
        l_source_train,
        l_source_test,
        l_target_train,
        l_target_test,
        algo="emd",
        reg=1,
        alpha=alpha,
    )
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")

pure_source, pure_target, test_source, test_target = (
    jdcoot.continuous_semisupervised_coot(
        source,
        target,
        source_test,
        target_test,
        l_source_train,
        l_source_test,
        l_target_train,
        l_target_test,
    )
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")
