import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
sys.path.append(os.path.abspath("src"))
import jdcoot


if __name__ == "__main__":
    from jdcoot.scenario import generate_data

    data = generate_data()

    pure, test = jdcoot.continuous_semisupervised_reference(*data)

    print(f"Pure Performance Reference : {pure}")
    print(f"Test Performance Reference : {test}")
