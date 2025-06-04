# JDCOOT : Algorithm for Transfer Learning in Incomparable Domains using Optimal Transport

Code : Lucas Offroy (Stage Ingénieur INSA)

Encadrement : Valérie Garès (INRIA)  et Chloé Friguet (UBS)

Support technique : Pierre Navaro (CNRS)

JDCOOT : **J**oint **D**istribution **CO**-**O**ptimal **T**ransport

Pour installer les dépendances python il faut utiliser [pixi](https://pixi.sh)

```bash
curl -fsSL https://pixi.sh/install.sh | bash
echo 'eval "$(pixi completion --shell bash)"' >> ~/.bashrc
source ~/.bashrc
```

Pour faire tourner l'un des scripts présents dans le dépôt (regression.py)

```bash
git clone https://github.com/otrecoding/jdcoot/
cd jdcoot
pixi install
pixi run python regression.py
```

## Numerical experiments

### alpha values

- classification
   + unsupervised : 0.661
   + semi-supervised : 3.335
   + partial : 2.875

- regression
   + unsupervised : 0.3
   + semi-supervised : 2.625
   + partial : 2.425

### mean shift between target and source

- classification and regression
    - `mean_shift = [0, 0.1, 0.2, 0.3, 0.4]`
    - unsupervised
        - balance = True
        - balance = False

### observed labels proportion variation

```python
prop_target = [0.1, 0.5, 0.9]
```

- classification and regression
- unsupervised
   - balance = True
   - balance = False
- semi-supervised
   - prop_target = [0, 0.02, 0.05, 0.07, 0.1, 0.12, 0.15]
     - balance = True
     - balance = False
   - prop_target = [0.1, 0.3, 0.5, 0.7, 0.9]
     - balance = True
     - balance = False
- partial 
   - prop_target = [0.1, 0.3, 0.5, 0.7, 0.9]
   - prop_target = [0.02, 0.05, 0.07, 0.1]


### sample size variation

- classification and regression
   - unsupervised
   - size = [10, 100, 500, 1000]


### sparse rate variation

- classification and regression
   - rates = [0.25, 0.5, 0.75, 1]
   - unsupervised

### correlation variation

- classification and regression
   - values = [0, 0.2, 0.5, 0.7, 1]
   - unsupervised

### proportion of observed covariates variation

- classification and regression
   - unsupervised
   - observed_covariates_proportion = `[0.2, 0.4, 0.6, 0.8]`

### OR variations

- classification
   - unsupervised
   - OR = `[0.2, 0.4, 0.6, 0.8]`

## R2 variations

- regression
   - unsupervised
   - values `[0.2, 0.4, 0.6, 0.8]`
   


