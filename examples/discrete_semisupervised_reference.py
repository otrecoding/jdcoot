import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

if __name__ == "__main__":
    from jdcoot.scenario import generate_data

    data = generate_data()

    perf_pure, perf_test = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.01
    )

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")

    perf_pure, perf_test = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.2
    )

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")

    perf_pure, perf_test = jdcoot.discrete_semisupervised_reference(
        *data, prop_target=0.3
    )

    print(f"Pure Performance Reference : {perf_pure}")
    print(f"Test Performance Reference : {perf_test}")
