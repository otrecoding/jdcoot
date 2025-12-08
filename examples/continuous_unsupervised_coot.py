import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot


if __name__ == "__main__":
    from jdcoot.scenario import generate_data

    data = generate_data()
    perf_pure, perf_test = jdcoot.continuous_unsupervised_coot(*data)

    print(f"Pure Performance COOT : {perf_pure}")
    print(f"Test Performance COOT : {perf_test}")
