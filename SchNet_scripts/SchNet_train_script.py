import numpy as np 
import os 

import importlib 
import SchNet_script 




dataset_directory_path = os.environ["DATASET_DIRECTORY_PATH"]
save_directory_path = os.environ["SAVE_DIRECTORY_PATH"]
os.makedirs(save_directory_path, exist_ok = True)

tae_scaling_factor = int(os.environ["TAE_SCALING_FACTOR"])
n_train = int(os.environ["N_TRAIN"])
n_val = int(os.environ["N_VAL"])
n_test = int(os.environ["N_TEST"])
seed = int(os.environ["SEED"])

batch_size = int(os.environ["BATCH_SIZE"])
n_hidden_channels = int(os.environ["N_HIDDEN_CHANNELS"])
n_filters = int(os.environ["N_FILTERS"])
n_interactions = int(os.environ["N_INTERACTIONS"])
n_gaussians = int(os.environ["N_GAUSSIANS"])
cutoff = float(os.environ["CUTOFF"])

lr = float(os.environ["LEARNING_RATE"])
n_epochs = int(os.environ["N_EPOCHS"])
checkpoint_epoch_interval = int(os.environ["CHECKPOINT_EPOCH_INTERVAL"])

xyz_coordinates_path = os.path.join(dataset_directory_path, "all_xyz_coordinates.npy")
nuclear_charges_path = os.path.join(dataset_directory_path, "all_nuclear_charges.npy")
tae_path = os.path.join(dataset_directory_path, "all_total_atomization_energy.npy")

all_xyz_coordinates = np.load(xyz_coordinates_path, allow_pickle = True)
all_nuclear_charges = np.load(nuclear_charges_path, allow_pickle = True)
all_tae = np.load(tae_path, allow_pickle = True) / tae_scaling_factor 

for filename in os.listdir(save_directory_path):
    file_path = os.path.join(save_directory_path, filename)
    if os.path.isfile(file_path): 
        os.remove(file_path)

train_indices, val_indices, test_indices = SchNet_script.train_val_test_splitting(
    len(all_tae), 
    n_train = n_train, 
    n_val = n_val, 
    n_test = n_test, 
    seed = seed, 
    test_indices = None 
)

np.save(
    os.path.join(save_directory_path, "train_indices.npy"), 
    train_indices
)
np.save(
    os.path.join(save_directory_path, "val_indices.npy"), 
    val_indices
)
np.save(
    os.path.join(save_directory_path, "test_indices.npy"), 
    test_indices
)

train_loader, val_loader, test_loader = SchNet_script.create_dataloaders(
    all_nuclear_charges, 
    all_xyz_coordinates, 
    all_tae, 
    train_indices, 
    val_indices, 
    test_indices, 
    batch_size = batch_size
)

schnet_model = SchNet_script.model_initializer(
    n_hidden_channels = n_hidden_channels, 
    n_filters = n_filters, 
    n_interactions = n_interactions, 
    n_gaussians = n_gaussians, 
    cutoff = cutoff, 
    device = "cuda"
)

SchNet_script.train_model(
    schnet_model, 
    train_loader, 
    val_loader, 
    lr = lr, 
    num_epochs = n_epochs, 
    device = "cuda", 
    checkpoint_epoch_interval = checkpoint_epoch_interval, 
    save_directory_path = save_directory_path
)