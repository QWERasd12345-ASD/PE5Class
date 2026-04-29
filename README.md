# PE5Class
# DNA Capsule Network Classifier

A DNA sequence classification project based on Capsule Network, supporting dual-channel input (FCGR images of original sequence and reverse complement), used for 5-class DNA sequence classification: prokaryotic virus, eukaryotic virus, prokaryote, eukaryote, and plasmid.

## Project Structure

main.py              # Main program entry, supports multiple running modes
config.py            # Configuration file, defines all hyperparameters and paths
models.py            # Capsule network model definition
trainer.py           # Trainer implementation
evaluator.py         # Evaluator implementation
data_loader.py       # Data loader
losses.py            # Loss function definition
utils.py             # Utility functions
CGR_utils.py         # CGR-related utility functions
generate_fcgr.py     # FCGR image generation script
generate_dna_labels.py # DNA label generation script
predict.py           # Prediction script
check.py             # Check script

## Features

- **Dual-channel Input**: Simultaneously processes FCGR images of original DNA sequence and its reverse complement
- **Capsule Network**: Capsule network architecture based on dynamic routing algorithm
- **Multi-mode Operation**: Supports training, testing, prediction, analysis, and data checking modes
- **Automatic GPU Acceleration**: Automatically detects and uses GPU for accelerated training
- **Early Stopping**: Prevents overfitting with early stopping mechanism
- **Learning Rate Scheduling**: Supports multiple learning rate scheduling strategies
- **Visualization**: Supports visualization of training process and prediction results

## Installation

pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn

## Data Preparation

### FCGR Image Format

Data should be in .npz format, containing the following keys:
- images: FCGR image data with shape (n_samples, 2, 64, 64)
- labels: Label data with shape (n_samples, num_classes) (one-hot encoding)

### Directory Structure

data/
└── fcgr_output/
    ├── train/
    │   └── fcgr.npz
    ├── val/
    │   └── fcgr.npz
    └── test/
        └── fcgr.npz

## Usage

### Training Mode

python main.py --mode train

### Testing Mode

python main.py --mode test --checkpoint outputs/models/best_model.pth

### Prediction Mode

python main.py --mode predict --checkpoint outputs/models/best_model.pth

### Analysis Mode

python main.py --mode analyze --checkpoint outputs/models/best_model.pth

### Data Check Mode

python main.py --mode data_check

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|--------|-------------|
| --mode | str | train | Running mode: train/test/predict/analyze/data_check |
| --checkpoint | str | None | Checkpoint path (for testing or resuming training) |
| --epochs | int | None | Number of training epochs (overrides config) |
| --batch_size | int | None | Batch size (overrides config) |
| --learning_rate | float | None | Learning rate (overrides config) |
| --visualize | bool | False | Whether to visualize results |
| --save_dir | str | None | Results save directory |
| --data_dir | str | None | Data directory |
| --train_data | str | None | Training data path |
| --val_data | str | None | Validation data path |
| --test_data | str | None | Test data path |

## Configuration

The config.py file contains the following main configurations:

### Data Configuration
- DATA_DIR: Data directory
- TRAIN_FCGR_PATH: Training data path
- VAL_FCGR_PATH: Validation data path
- TEST_FCGR_PATH: Test data path

### Model Configuration
- NUM_CLASSES: Number of classes (default 5)
- INPUT_SHAPE: Input image shape (1, 64, 64)
- PRIMARY_CAPSULES: Number of primary capsules
- PRIMARY_DIM: Primary capsule dimension
- DIGIT_DIM: Digit capsule dimension
- NUM_ROUTING: Dynamic routing iterations

### Training Configuration
- BATCH_SIZE: Batch size
- EPOCHS: Number of training epochs
- LEARNING_RATE: Learning rate
- WEIGHT_DECAY: Weight decay

### Class Mapping

| Index | Class Name | Description |
|-------|------------|-------------|
| 0 | prokaryotic_virus | Prokaryotic Virus |
| 1 | eukaryotic_virus | Eukaryotic Virus |
| 2 | prokaryote | Prokaryote |
| 3 | eukaryote | Eukaryote |
| 4 | plasmid | Plasmid |

## Model Architecture

The capsule network used in this project consists of the following components:

1. **Feature Extractors** (Dual-channel)
   - Original sequence feature extractor
   - Reverse complement feature extractor

2. **Primary Capsule Layer**
   - Converts convolutional features to initial capsule vectors

3. **Digit Capsule Layer**
   - Uses dynamic routing algorithm for feature combination

4. **Decoder**
   - Reconstructs FCGR images from capsule vectors

## Output Files

After training, the output directory structure is as follows:

outputs/
├── models/
│   └── best_model.pth          # Best model weights
├── logs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── training_history.png # Training history plot
│       ├── training.log         # Training log
│       └── visualizations/      # Visualization results
└── checkpoints/
    └── checkpoint_epoch_X.pth   # Training checkpoints

