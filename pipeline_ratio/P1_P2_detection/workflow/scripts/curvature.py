import numpy as np

pulses = np.load(snakemake.input["files"], allow_pickle=True)

curves = []
for pulse in pulses:
    diff1 = np.gradient(-pulse*100)
    diff2 = np.gradient(diff1)
    curve = diff2/(1 + diff1**2)**(3/2)
    curves.append(curve)

np.save(snakemake.output['curvature'], np.stack(curves))