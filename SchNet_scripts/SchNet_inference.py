import numpy as np 
from tqdm import tqdm 
import os 
import torch 
from torch_geometric.nn import SchNet 
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, InMemoryDataset




def model_initializer(
        n_hidden_channels = 250, 
        n_filters = 250, 
        n_interactions = 5,
        n_gaussians = 50, 
        cutoff = 10.0, 
        device = "cuda"
):
    """
    Recommended to have n_hidden_channels = n_filters
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using:", device)

    model = SchNet(
        hidden_channels = n_hidden_channels, 
        num_filters = n_filters, 
        num_interactions = n_interactions, 
        num_gaussians = n_gaussians, 
        cutoff = cutoff 
    )
    model.to(device)
    return model 


class TestDataset(InMemoryDataset):
    def __init__(self, zs, positions, transform=None):
        self.zs = zs
        self.positions = positions
        super().__init__(".", transform)
        self.data, self.slices = self.collate([
            self._to_data(z, pos) 
            for z, pos in zip(zs, positions)
        ])

    def _to_data(self, z, pos):
        return Data(
            z = torch.tensor(z, dtype = torch.long),
            pos = torch.tensor(pos, dtype = torch.float)
        )


def model_inference(
        checkpoint_path, 
        nuclear_charges, 
        xyz_coordinates, 
        n_hidden_channels = 250, 
        n_filters = 250, 
        n_interactions = 5, 
        n_gaussians = 50, 
        cutoff = 10.0, 
        device = "cuda", 
        batch_size = 1000, 
        tae_scaling_factor = 25 
):
    """
    - Returns total atomization energy predictions with units of kcal/mol
    - 'nuclear_charges' is an array of nuclear charges
    - 'xyz_coordinates' is an array containing arrays of nuclear coordinates
      corresponding to 'nuclear_charges' in units of angstrom
    """
    
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else: 
        device = torch.device("cpu")
    
    model = model_initializer(
        n_hidden_channels = n_hidden_channels, 
        n_filters = n_filters, 
        n_interactions = n_interactions, 
        n_gaussians = n_gaussians, 
        cutoff = cutoff, 
        device = device 
    )
    model.load_state_dict(
        torch.load(checkpoint_path)["model_state_dict"]
    )
    model.eval()

    test_dataset = TestDataset(nuclear_charges, xyz_coordinates)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    predictions = [] 
    with torch.no_grad(): 
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            predictions.append(out.cpu())
    predictions = torch.cat(predictions, dim = 0)
    return predictions.numpy().flatten() * tae_scaling_factor 