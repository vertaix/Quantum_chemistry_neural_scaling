#!/bin/bash 
#SBATCH --job-name=learning_curve_QM9_G4MP2_10000_train_500_basis_3_interactions
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16 
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=120:00:00
#SBATCH --output=Output.%j.out
#SBATCH --error=Output.%j.err

N_SAMPLES=10000

export DATASET_DIRECTORY_PATH="/input/your/example/dataset_directory_path/Datasets/QM9_G4MP2"
export SAVE_DIRECTORY_PATH="/input/your/example/save_directory_path/QM9_G4MP2/Train_10000"

export TAE_SCALING_FACTOR=25
export N_TRAIN=${N_SAMPLES}
export N_VAL=1000
export N_Test=1000
export SEED=21

export BATCH_SIZE=50
export N_HIDDEN_CHANNELS=500 
export N_FILTERS=500
export N_INTERACTIONS=3
export N_GAUSSIANS=50
export CUTOFF=5.0

export LEARNING_RATE=0.0005
export N_EPOCHS=1000
export CHECKPOINT_EPOCH_INTERVAL=5

source "/input/your/path/to/venv/bin/activate"
srun /input/your/path/to/venv/bin/python /input/your/path/to/SchNet_train_script.py