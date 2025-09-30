# Are Neural Scaling Laws Leading Quantum Chemistry Astray?

This repository contains the code and data utilized in the experiments conducted in this paper. Specifically:

## Virtual Environments

"Virtual_environments" contains Python packages utilized for the foundation models and training SchNet models. To install the packages, activate your virtual environment and run

```
pip install -r name_of_requirements.txt
```

We recommend the use of "venv_aimnet2_requirements.txt" for AIMNet2, "venv_meta_requirements.txt" for META's models, "venv_orbital_materials_requirements.txt" for the Orb v3 model, and "venv_schnet_train_requirements.txt" for training the SchNet models as we have done in our work.


## Training and Using SchNet Models

"SchNet_scripts" contains Python code to implement the scaling experiments in our work using the SchNet architecture. We provide an example bash script that executes the training. Note the following:
- "N_SAMPLES" sets number of training samples.
- "DATASET_DIRECTORY_PATH" sets path to the directory containing nuclear charges, nuclear coordinates, and atomization energies for a given dataset (e.g. GDB-9-G4(MP2) or VQM24).
- "SAVE_DIRECTORY_PATH" sets path to directory to save the results.
- "TAE_SCALING_FACTOR" sets (inverse) scaling factor for the atomization energies (in units of kcal/mol). For example, we used 25 to scale by 1/25 so that kcal/mol is of similar magnitude to eV.
- "N_VAL" sets number of validation samples.
- "N_TEST" sets number of test samples.
- "SEED" sets seed for reproducibility purposes.
- "BATCH_SIZE" sets batch size.
- "N_HIDDEN_CHANNELS", "N_FILTERS", "N_INTERACTIONS", "N_GAUSSIANS", "CUTOFF" set hyperparameters for SchNet architecture.
- "LEARNING_RATE" sets learning rate.
- "N_EPOCHS" sets maximum number of training epochs.
- "CHECKPOINT_EPOCH_INTERVAL" sets how frequently model checkpoints are saved.

Once a model is trained and saved, it can be used for inference. To do so, first import 

```python
import numpy as np
from tqdm import tqdm
import os
import torch
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
```

and predict energies from provided arrays of nuclear_charges and xyz coordinates, along with the exact hyperparameters used for the trained model:

```python
predicted_energy = model_inference(
     model_checkpoint_path,
     nuclear_charges_array,
     xyz_coordinates_array,
     n_hidden_channels = N_HIDDEN_CHANNELS,
     n_filters = N_FILTERS,
     n_interactions = N_INTERACTIONS,
     n_gaussians = N_GAUSSIANS,
     cutoff = CUTOFF,
     device = "cuda",
     batch_size = BATCH_SIZE,
     tae_scaling_factor = TAE_SCALING_FACTOR
)
```


## Using Foundation Models

"Foundation_models_scripts" contains Python code for utilizing the foundation machine-learned interatomic potentials. 
For example, to use UMA-S-1.1, first import

```python
from ase import Atoms
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
```

and initialize the model by calling 

```python
model = uma_s_1p1_model_initializer(device = "cpu")
```

and predict energies from provided arrays of element symbols and xyz coordinates, along with total charge and spin multiplicity, by using

```python
predicted_total_energy = predict_energy(
     model,
     element_symbols_array,
     xyz_coordinates_array,
     charge = 0,
     spin_multiplicity = 1
)
```


## Training Data

Refer to the following Zenodo repository for nuclear charges, nuclear coordinates, atomization energies used as training data and the model checkpoints from scaling experiments: https://zenodo.org/records/17202891
