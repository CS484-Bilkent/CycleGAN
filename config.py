import torch
from torch.vision.transforms import v2

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
LEARNING_RATE = 1e-5  # TODO: try with 2e-4

# From the paper:
# The identity mapping loss of weight 0.5λ was used.
# We set λ = 10.
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10

NUM_WORKERS = 4

NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_A = "gen_a.pth"
CHECKPOINT_GEN_B = "gen_b.pth"
CHECKPOINT_CRITIC_A = "critic_a.pth"
CHECKPOINT_CRITIC_B = "critic_b.pth"
