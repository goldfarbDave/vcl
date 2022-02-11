from sre_parse import FLAGS
import torch
import numpy as np
import flags
import random

def seed_everything():
    torch.manual_seed(flags.FLAGS['seed'])
    torch.cuda.manual_seed(flags.FLAGS['seed'])
    torch.cuda.manual_seed_all(flags.FLAGS['seed'])
    np.random.seed(flags.FLAGS['seed'])
    random.seed(flags.FLAGS['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True