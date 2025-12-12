import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

if __name__ == "__main__":
    from jdcoot.scenario import generate_data

    data = generate_data()

    pure_source, pure_target, test_source, test_target = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.01
    )

    print(f"Pure performance COOT on source : {pure_source} ")
    print(f"Pure performance COOT on target : {pure_target} ")
    print(f"Test performance COOT on source : {test_source} ")
    print(f"Test performance COOT on target : {test_target} ")

    pure_source, pure_target, test_source, test_target = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.2
    )

    print(f"Pure performance COOT on source : {pure_source} ")
    print(f"Pure performance COOT on target : {pure_target} ")
    print(f"Test performance COOT on source : {test_source} ")
    print(f"Test performance COOT on target : {test_target} ")

    pure_source, pure_target, test_source, test_target = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.3
    )

    print(f"Pure performance COOT on source : {pure_source} ")
    print(f"Pure performance COOT on target : {pure_target} ")
    print(f"Test performance COOT on source : {test_source} ")
    print(f"Test performance COOT on target : {test_target} ")
