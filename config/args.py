import argparse
import config.config as config

parser = argparse.ArgumentParser(description="Unofficial CycleGAN implementation using PyTorch")

parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--device", type=str, default=config.DEVICE)
parser.add_argument("--train-dir", type=str, default=config.TRAIN_DIR)
parser.add_argument("--test-dir", type=str, default=config.TEST_DIR)
parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
parser.add_argument("--lambda-identity", type=float, default=config.LAMBDA_IDENTITY)
parser.add_argument("--lambda-cycle", type=float, default=config.LAMBDA_CYCLE)
parser.add_argument("--num-workers", type=float, default=config.NUM_WORKERS)
parser.add_argument("--num-epochs", type=float, default=config.NUM_EPOCHS)
parser.add_argument("--load-checkpoints", type=bool, default=config.LOAD_CHECKPOINTS)
parser.add_argument("--save-checkpoints", type=bool, default=config.SAVE_CHECKPOINTS)
parser.add_argument("--save-checkpoints-epoch", type=int, default=config.SAVE_CHECKPOINTS_EPOCH)
