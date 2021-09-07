import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import json
import time
import pathlib
import os
import pickle
from tqdm import tqdm
import random
import pandas as pd
import pdb
dirname = str(pathlib.Path(__file__).parent.absolute())

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
