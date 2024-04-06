import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# device='cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
torch.manual_seed(1337)
BLOCK=8
BS=4
EPOCHS=10000
EVAL_STEPS=10
N_EMBS=32
LR=1e-3