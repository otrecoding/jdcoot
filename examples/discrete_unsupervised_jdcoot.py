import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot

if __name__ == "__main__":
    from jdcoot.scenario import generate_data

    data = generate_data()

    pure_source, pure_target, test_source, test_target = (
        jdcoot.discrete_unsupervised_jdcoot(*data)
    )

    print(f"Pure performance source : {pure_source}")
    print(f"Pure performance target : {pure_target}")
    print(f"Test performance source : {test_source}")
    print(f"Test performance target : {test_target}")
