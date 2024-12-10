# JDCOOT : Algorithm for Transfer Learning in Incomparable Domains using Optimal Transport

Code stage Lucas Offroy

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
