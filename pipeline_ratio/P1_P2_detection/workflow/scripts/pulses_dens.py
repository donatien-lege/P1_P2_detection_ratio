import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torch

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def curvature(pulse):
    curves = []
    for line in pulse: 
        diff1 = np.array(np.gradient(-line * 100))
        diff2 = np.gradient(diff1)
        curve = diff2/(1 + diff1**2)**(3/2)
        curve = curve/max(curve)
        curve = np.clip(curve, 0, 1)
        curves.append(curve)
    return np.stack(curves)

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
        return self.pulses[idx, :], self.annots[idx, :]
    