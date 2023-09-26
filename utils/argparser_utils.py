import os
import torch
import random
import argparse
import numpy as np

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path_check(args):
    # Preprocessing Path Checking
    if not os.path.exists(args.preprocess_path):
        os.makedirs(args.preprocess_path)

    # Model Checkpoint Path Checking
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    # Testing Results Path Checking
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)