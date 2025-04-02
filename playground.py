import numpy as np
import matplotlib
import scipy
import random
import time
import torch
from matplotlib import pyplot as plt


t1 = time.time()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch is using {device}")

# Create a simple tensor and perform a basic operation
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Tensor addition:", a + b)

t2 = time.time()

print(t2-t1)