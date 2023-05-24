import torch
from torch import nn
from utils import PulsesDataset
from torch.utils.data import DataLoader
import models
from torch.utils.data import random_split
import torch.optim as optim
import numpy as np
import pandas as pd

#Params
BATCH = snakemake.config['batch']
SPLIT = snakemake.config['split']
LR = snakemake.config['l_rate']
MODEL = eval(f"models.{snakemake.wildcards['nn']}()")
EPOCHS = snakemake.config['epochs']
PULSES = snakemake.params['folder_pulse']
ANNOTS = snakemake.params['folder_annot']

#Dataset
pulses = PulsesDataset(PULSES, ANNOTS)
loader = DataLoader(pulses,
                    batch_size=BATCH, 
                    shuffle=True,
                    generator=torch.Generator().manual_seed(0))
lim = int(len(pulses) / BATCH * SPLIT)
train_set, val_set = random_split(loader, [lim, len(loader) - lim],
                                  generator=torch.Generator().manual_seed(0))

#Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = MODEL.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

epochs = []
losses = []

for epoch in range(EPOCHS): 
    
    for i, data in enumerate(train_set.dataset):
        
        inputs, labels = data
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(device))
        loss = criterion(outputs.to(device), labels.to(device))
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 9:
        with torch.no_grad():
            running_loss = []
            for i, data in enumerate(val_set.dataset):
                inputs, labels = data

                # forward + backward + optimize
                outputs = net(inputs.float().to(device))
                loss = criterion(outputs.to(device), labels.to(device))
                # print statistics
                running_loss.append(float(loss.detach()))
            epochs.append(epoch+1)
            losses.append(round(np.mean(running_loss), 5))
            print(snakemake.wildcards['nn'])
            print("epoch #", epoch+1)

df = pd.DataFrame({'epoch': epochs, 'loss': losses})
df.to_csv(snakemake.output['loss'], index=False)
torch.save(net.state_dict(), snakemake.output['model'])