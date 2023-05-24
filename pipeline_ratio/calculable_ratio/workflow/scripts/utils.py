import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class PulsesDataset(Dataset):
    
    def __init__(self, pul, ann):
        
        fp = glob.glob(f"{pul}/*")
        fa = glob.glob(f"{ann}/*")
        
        pulses = [np.load(p, allow_pickle=True) for p in fp]
        annots = [np.load(a, allow_pickle=True) for a in fa]
        self.pulses = np.concatenate(pulses)
        self.annots = np.concatenate(annots).astype(int)
        self.pulses = torch.from_numpy(self.pulses).float()
        self.annots = torch.from_numpy(self.annots).float()

    def __len__(self):
        return len(self.pulses)
    
    def __getitem__(self, idx):
        return self.pulses[idx, :], self.annots[idx]
    