import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath("src"))
import jdcoot

prop_source = 0.5
prop_target = 0.005

source, target, source_test, target_test = jdcoot.generate_data()
n_target = len(target.Z)
n_source = len(source.Z)
l_source_train, l_source_test = train_test_split(
    np.arange(n_source), train_size=prop_source
)
l_target_train, l_target_test = train_test_split(
    np.arange(n_target), train_size=prop_target
)

pure_source, pure_target, test_source, test_target = jdcoot.discrete_partial_coot(
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

pure_source, pure_target, test_source, test_target = jdcoot.discrete_partial_jdcoot(
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

pure_source, pure_target, test_source, test_target = jdcoot.discrete_partial_reference(
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

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_semisupervised_coot(
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
    )
)

print(f"Pure performance on source : {pure_source} ")
print(f"Pure performance on target : {pure_target} ")
print(f"Test performance on source : {test_source} ")
print(f"Test performance on target : {test_target} ")

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_semisupervised_reference(
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

print(f"Pure performance COOT on source : {pure_source} ")
print(f"Pure performance COOT on target : {pure_target} ")
print(f"Test performance COOT on source : {test_source} ")
print(f"Test performance COOT on target : {test_target} ")

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_semisupervised_reference(
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

print(f"Pure performance COOT on source : {pure_source} ")
print(f"Pure performance COOT on target : {pure_target} ")
print(f"Test performance COOT on source : {test_source} ")
print(f"Test performance COOT on target : {test_target} ")

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_semisupervised_reference(
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

print(f"Pure performance COOT on source : {pure_source} ")
print(f"Pure performance COOT on target : {pure_target} ")
print(f"Test performance COOT on source : {test_source} ")
print(f"Test performance COOT on target : {test_target} ")

pure_source, pure_target, test_source, test_target = jdcoot.discrete_unsupervised_coot(
    source,
    target,
    source_test,
    target_test,
    l_source_train,
    l_source_test,
    l_target_train,
    l_target_test,
)

print(f"Pure performance source : {pure_source}")
print(f"Pure performance target : {pure_target}")
print(f"Test performance source : {test_source}")
print(f"Test performance target : {test_target}")

pure_source, pure_target, test_source, test_target = (
    jdcoot.discrete_unsupervised_jdcoot(
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

print(f"Pure performance source : {pure_source}")
print(f"Pure performance target : {pure_target}")
print(f"Test performance source : {test_source}")
print(f"Test performance target : {test_target}")
