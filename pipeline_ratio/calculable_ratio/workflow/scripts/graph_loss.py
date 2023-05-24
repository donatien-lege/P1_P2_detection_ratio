import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

files = snakemake.input['files']
dfs = [pd.read_csv(file, index_col='epoch') for file in files]
df = pd.concat(dfs, axis=1)
df.columns = [f.split('/')[-1].split('.')[0] for f in files]
df.plot()
plt.savefig(snakemake.output['loss'])