# JDCOOT : A Joint Distribution Co-Optimal Transport Approach for Adapting Models to Heterogeneous Domains

Code : Lucas Offroy (Intern Engineer INSA)

Advisors : Valérie Garès (INRIA) and Chloé Friguet (UBS)

Maintenance : Pierre Navaro (CNRS)

This work addresses a fundamental challenge in modern statistical learning: adapting mod-
els to heterogeneous domains, where source and target data are characterised by different
feature spaces and underlying distributions. We introduce Joint Distribution Co-Optimal
Transport (JDCOOT), a domain adaptation algorithm that leverages optimal transport
to align the joint feature-label distributions of distinct domains, enabling effective knowl-
edge transfer across heterogeneous domains. 

## Installation

Use [pixi](https://pixi.sh) to run the code

```bash
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ~/.bashrc
source ~/.bashrc
```

All run scripts are in the `examples` directory:

```bash
git clone https://github.com/otrecoding/jdcoot/
cd jdcoot
pixi install
pixi run python examples/discrete_partial_jdcoot.py
```

## Numerical experiments

For all expermiments we use three methods `reference`, `coot` and `jdcoot` on two datasets (train and test).
The "pure" performance is the accuracy and the train dataset and the "test" performance is the prediction accuracy
using the same model on another dataset.

Input : 

learning data, 2 dataframes (source and target) with the following format :

$$
| X_1 | ... | X_d | Y | Z |
$$

- $X_i$ the ith observed covariate,
- $Z$ the discrete objective variable for classification analysis

test data, 2 dataframes (test_source and test_target)

supervision : 

- `unsupervised` : none of the observations of train target are labelled and all source observations are labelled
- `semi-supervised` : `prop_target` of the observations of train target are labelled and all train source observations are labelled
- `partial` : `prop_target` of the observations of train target are labelled and `prop_source` of the observations of train source are labelled
