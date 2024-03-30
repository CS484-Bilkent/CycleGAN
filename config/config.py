import torch

"""
For all the experiments, we set λ = 10 in Equation 3. 
We use the Adam solver [26] with a batch size of 1. 
All networks were trained from scratch with a learning rate of 0.0002. 
We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
TEST_DIR = "dataset/test"
BATCH_SIZE = 1  # Adjust as needed
LEARNING_RATE = 2e-4  # TODO: try with 1e-5
ADJUST_LEARNING_RATE = 2e-4  # TODO: try with 1e-5

# From the paper:
# The identity mapping loss of weight 0.5λ was used.
# We set λ = 10.
LAMBDA_IDENTITY = 10
LAMBDA_CYCLE = 10

NUM_WORKERS = 4

NUM_EPOCHS = 10
LOAD_CHECKPOINTS = True
LOAD_CHECKPOINTS_PATH = "checkpoints"
SAVE_CHECKPOINTS = True

SAVE_CHECKPOINTS_EPOCH = 1
