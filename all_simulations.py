import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from continuous_partial_coot import continuous_partial_coot
from continuous_partial_jdcoot import continuous_partial_jdcoot
from continuous_partial_reference import continuous_partial_reference
from continuous_semisupervised_coot import continuous_semisupervised_coot
from continuous_semisupervised_jdcoot import continuous_semisupervised_jdcoot
from continuous_semisupervised_reference import continuous_semisupervised_reference
from continuous_unsupervised_coot import continuous_unsupervised_coot
from continuous_unsupervised_jdcoot import continuous_unsupervised_jdcoot
from discrete_partial_coot import discrete_partial_coot
from discrete_partial_jdcoot import discrete_partial_jdcoot
from discrete_partial_reference import discrete_partial_reference
from discrete_semisupervised_coot import discrete_semisupervised_coot
from discrete_semisupervised_jdcoot import discrete_semisupervised_jdcoot
from discrete_semisupervised_reference import discrete_semisupervised_reference
from discrete_unsupervised_coot import discrete_unsupervised_coot
from discrete_unsupervised_jdcoot import discrete_unsupervised_jdcoot
from scenario import generate_data


data = generate_data()

print("continuous")
print("\t partial ")
print("\t\t coot")
pure, test =  continuous_partial_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  continuous_partial_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t reference")
pure, test =  continuous_partial_reference(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t semi-supervised ")
print("\t\t coot")
pure, test =  continuous_semisupervised_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  continuous_semisupervised_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t reference")
pure, test =  continuous_semisupervised_reference(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t unsupervised ")
print("\t\t coot")
pure, test =  continuous_unsupervised_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  continuous_unsupervised_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("discrete")
print("\t partial ")
print("\t\t coot")
pure, test =  discrete_partial_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  discrete_partial_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t reference")
pure, test =  discrete_partial_reference(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t semi-supervised ")
print("\t\t coot")
pure, test =  discrete_semisupervised_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  discrete_semisupervised_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t reference")
pure, test =  discrete_semisupervised_reference(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t unsupervised ")
print("\t\t coot")
pure, test =  discrete_unsupervised_coot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
print("\t\t jdcoot")
pure, test =  discrete_unsupervised_jdcoot(*data)
print(f"pure, test = {pure:7.3f}, {test:7.3f}")
