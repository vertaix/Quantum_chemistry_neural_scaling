import numpy as np 
from tqdm import tqdm 
import os 
import torch 
from torch_geometric.nn import SchNet 
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, InMemoryDataset 
import torch.nn.utils as utils 
import torch.nn as nn 




def train_val_test_splitting(
        dataset_size, 
        n_train = 5000, 
        n_val = 1000, 
        n_test = 1000, 
        seed = 21, 
        test_indices = None
):
    dataset_size = int(dataset_size)
    n_train = int(n_train)
    n_val = int(n_val)
    n_test = int(n_test)
    rng = np.random.default_rng(seed)

    if test_indices is not None: 
        n_test = int(len(test_indices))
    else: 
        test_indices = rng.choice(
            range(dataset_size), 
            size = n_test, 
            replace = False 
        )
        n_test = int(len(test_indices))
    assert int(n_train+n_val+n_test)<=dataset_size, "Total train, validation, test split sizes are too large."

    train_val_indices = rng.choice(
        np.delete(range(dataset_size), test_indices), 
        size = n_train + n_val, 
        replace = False
    )
    rng.shuffle(train_val_indices)
    train_indices = train_val_indices[:n_train]
    val_indices = train_val_indices[n_train:]

    return train_indices, val_indices, test_indices 


class CreateDataset(InMemoryDataset):
    def __init__(
            self, 
            all_nuclear_charges, 
            all_positions, 
            all_labels, 
            transform = None 
    ):
        super().__init__(None, transform)
        self.data_list = [] 

        for nuclear_charges, positions, labels in zip(
            all_nuclear_charges, 
            all_positions, 
            all_labels
        ):
            nuclear_charges_tensor = torch.tensor(nuclear_charges, dtype = torch.long)
            positions_tensor = torch.tensor(positions, dtype = torch.float)
            labels_tensor = torch.tensor([labels], dtype = torch.float)

            self.data_list.append(Data(
                z = nuclear_charges_tensor, 
                pos = positions_tensor, 
                y = labels_tensor 
            ))
        self.data, self.slices = self.collate(self.data_list)

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    

def create_dataloaders(
        all_nuclear_charges, 
        all_xyz_coordinates, 
        all_labels, 
        train_indices, 
        val_indices, 
        test_indices, 
        batch_size = 10
):
    train_dataset = CreateDataset(
        all_nuclear_charges[train_indices], 
        all_xyz_coordinates[train_indices], 
        all_labels[train_indices]
    )
    val_dataset = CreateDataset(
        all_nuclear_charges[val_indices], 
        all_xyz_coordinates[val_indices], 
        all_labels[val_indices]
    )
    test_dataset = CreateDataset(
        all_nuclear_charges[test_indices], 
        all_xyz_coordinates[test_indices], 
        all_labels[test_indices]
    )

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size)

    return train_loader, val_loader, test_loader 


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


def checkpoint_epochs_indices(num_epochs, interval = 5):
    result = list(range(0, int(num_epochs), interval))
    if result[-1] != int(num_epochs) - 1:
        result.append(int(num_epochs) - 1)
    return result 


def train_model(
        model, 
        train_loader, 
        val_loader, 
        lr = 1e-4, 
        num_epochs = 20, 
        device = "cuda", 
        checkpoint_epoch_interval = 5, 
        save_directory_path = None, 
        early_stop_patience = 50, 
        scheduler_factor = 0.5, 
        scheduler_patience = 20, 
        warmup_epochs = 10 
):
    if device == "cuda" and torch.cuda.is_available(): 
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss_fn = torch.nn.MSELoss() 

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0 
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = warmup_lambda)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = "min", 
        factor = scheduler_factor, 
        patience = scheduler_patience 
    )
    best_val_loss = float("inf")
    patience_counter = 0 

    checkpoint_indices = checkpoint_epochs_indices(num_epochs, interval = checkpoint_epoch_interval)
    last_model_checkpoint_path = os.path.join(save_directory_path, "last_model.ckpt")
    best_model_checkpoint_path = os.path.join(save_directory_path, "best_model.ckpt")
    train_loss_history = [] 
    val_loss_history = [] 

    for epoch in tqdm(range(int(num_epochs))):
        model.train() 
        total_train_loss = 0 
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.z, data.pos, data.batch)
            loss = loss_fn(pred.view(-1), data.y.view(-1))
            loss.backward() 
            utils.clip_grad_norm_(model.parameters(), 1000)
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs 
        train_loss = total_train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        model.eval() 
        total_val_loss = 0 
        with torch.no_grad(): 
            for data in val_loader: 
                data = data.to(device)
                pred = model(data.z, data.pos, data.batch)
                loss = loss_fn(pred.view(-1), data.y.view(-1))
                total_val_loss += loss.item() * data.num_graphs 
            val_loss = total_val_loss / len(val_loader.dataset)
            val_loss_history.append(val_loss)

        if epoch < warmup_epochs: 
            warmup_scheduler.step() 
        else: 
            plateau_scheduler.step(val_loss)

        if int(epoch) in checkpoint_indices: 
            torch.save(
                {
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(), 
                    "train_loss": train_loss, 
                    "val_loss": val_loss
                }, 
                last_model_checkpoint_path
            )

        if val_loss < best_val_loss: 
            best_val_loss = val_loss 
            torch.save(
                {
                    "epoch": epoch, 
                    "model_state_dict": model.state_dict(), 
                    "optimizer_state_dict": optimizer.state_dict(), 
                    "train_loss": train_loss, 
                    "val_loss": val_loss 
                }, 
                best_model_checkpoint_path
            )
            patience_counter = 0 
        else: 
            patience_counter += 1 
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break 
        
    train_loss_history = np.array(train_loss_history)
    val_loss_history = np.array(val_loss_history)

    np.save(
        os.path.join(save_directory_path, "train_loss_history.npy"), 
        train_loss_history
    )
    np.save(
        os.path.join(save_directory_path, "val_loss_history.npy"), 
        val_loss_history
    )