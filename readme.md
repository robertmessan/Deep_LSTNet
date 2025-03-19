# Deep_LSTNet

## Description:
Ce projet s'inscrit dans le cadre du cours introduction au deep learning. Nous avons implémenté l'architecture de réseau de neurones proposée dans l'article https://arxiv.org/pdf/1703.07015, puis nous l'avons enrichie avec les techniques de recherche d'hyperparamètres comme le early stopping et le Learning rate schedulling.
## Prérequis:
Il faut disposer d'un GPU cuDA ou ses équivalent pour exécuter cette architecture, elle est spécialement codée en utilisant cuDA.
Certains packages comme ```bash tensorboard``` et ```bash torch``` doivent être installés sur votre machine
## Structure du projet:

```txt
.
├── models
│   ├── LSTNet.py
│   ├── __init__.py
│   ├── early_stopping.py
│   └── exchange_rate.pt
├── runs
│   ├── air_quality
│   ├── electricity
│   ├── exp1
│   ├── solar
│   ├── stock
│   └── traffic
├── .gitignore
├── Optim.py
├── air_quality.sh
├── ele.sh
├── main.py
├── readme.md
├── solar.sh
├── stock.sh
├── traffic.sh
└── utils.py

## Installation & Exécution

1. **Clonez ou téléchargez** ce dépôt :
   ```bash
   git clone [https://github.com/votre_compte/votre_projet.git](https://github.com/robertmessan/Deep_LSTNet.git)
2. Activez votre GPU (si disponible et nécessaire) en configurant votre environnement (ex. : CUDA ou cuDNN).

3. Rendez vos scripts exécutables, par exemple :

```bash
chmod +x stock.sh

4. Exécutez le script souhaité, par exemple :
```bash
./stock.sh

5. Les courbes d’entraînement (logs, checkpoints, etc.) seront générées dans le dossier runs ou dans un autre dossier précisé dans le script.
Pour visualiser les résultats avec TensorBoard, exécutez la commande suivante :
```bash
tensorboard --logdir runs

Vous pouvez visualiser vos courbes sur votre navigateur via localhost
