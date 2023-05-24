import torch
import models
import numpy as np

model = eval(f"models.{snakemake.wildcards['nn']}()")
model.load_state_dict(torch.load((snakemake.input['model'])))
arr = np.load(snakemake.input['pulse'], allow_pickle=True)

pulses = torch.from_numpy(arr)
res = model(pulses.float()).detach().numpy()

np.save(snakemake.output['pred'], res)
