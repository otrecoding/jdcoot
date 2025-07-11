# JDCOOT : Algorithm for Transfer Learning in Incomparable Domains using Optimal Transport

Code : Lucas Offroy (Intern Engineer INSA)

Advisors : Valérie Garès (INRIA) and Chloé Friguet (UBS)

Maintenance : Pierre Navaro (CNRS)

JDCOOT : **J**oint **D**istribution **CO**-**O**ptimal **T**ransport

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

For all expermiments we use three methods `reference`, `coot` and `jdcoot`

Input : 

learning data, 2 dataframes (source and target) with the following format :

$$
| X_1 | ... | X_d | Y | Z |
$$

- $X_i$ the ith observed covariate,
- $Y$ the continuous objective variable for regression analysis,
- $Z$ the discrete objective variable for classification analysis

test data, 2 dataframes (test_source and test_target)

supervision : 

- `unsupervised` : none of the observations of target are labelled and all source observations are labelled
- `semi-supervised` : `prop_target` of the observations of target are labelled and all source observations are labelled
- `partial` : `prop_target` of the observations of target are labelled and `prop_source` of the observations of source are labelled

### alpha values

Hyper parameter of the problem formulation as in the paper. 

- classification
   + unsupervised : 0.661
   + semi-supervised : 3.335
   + partial : 2.875

- regression
   + unsupervised : 0.3
   + semi-supervised : 2.625
   + partial : 2.425

### Mean shift between target and source

- classification and regression
    - `mean_shift = [0, 0.1, 0.2, 0.3, 0.4]`
    - unsupervised

### Observed labels proportion variation

- classification and regression
   - unsupervised
   - semi-supervised
      + `prop_target = [0.02, 0.05, 0.07, 0.1, 0.12, 0.15, 0.3, 0.5, 0.7, 0.9]`
   - partial 
      + `prop_source = [0.02, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9]`
      + `prop_target = [0.02, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9]`

### Sample size variation

- classification and regression
   - unsupervised
   - `size = [10, 100, 500, 1000]`

### Sparse rate variation

- classification and regression
   - `rates = [0.25, 0.5, 0.75, 1]`
   - unsupervised

### Auto-correlation coefficients variation

- classification and regression
   - `values = [0, 0.2, 0.5, 0.7, 1]`
   - unsupervised

### Proportion of observed covariates variation

- classification and regression
   - unsupervised
   - `observed_covariates_proportion = [0.2, 0.4, 0.6, 0.8]`

### Odds ratio coefficients variation

- classification
   - unsupervised
   - `OR = [0.2, 0.4, 0.6, 0.8]`

## R2 variations

- regression
   - unsupervised
   - `values = [0.2, 0.4, 0.6, 0.8]`
