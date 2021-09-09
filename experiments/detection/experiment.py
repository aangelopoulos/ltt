# import some common libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import os, json, cv2, random, sys, traceback
from utils import *

import multiprocessing as mp

import pickle as pkl
from tqdm import tqdm
import pdb

if __name__ == "__main__":
    fix_randomness(seed=0)
    lambda1s = torch.linspace(0.5,1,5) # Top score threshold
    lambda2s = torch.linspace(0,1,10) # Segmentation threshold
    lambda3s = torch.tensor([0.9,0.925,0.95,975,0.99,0.995,0.999,0.9995,1]) # APS threshold
    with torch.no_grad():
	# Load cache
        with open('./.cache/loss_tables.pt', 'rb') as f:
            loss_tables = torch.load(f)
        
        pdb.set_trace()
        print("Done!")
