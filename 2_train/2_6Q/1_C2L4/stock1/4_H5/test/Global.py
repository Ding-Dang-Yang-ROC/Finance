import numpy as np
import torch

Stock=0
Horizon=5
N_Qubit=6
N_Lay1=4
S_Ker1=4
N_par1=S_Ker1*S_Ker1
N_par2=4
N_Ite=1000
learning_rate=1e-5
Batch=100

Dtype=torch.float
