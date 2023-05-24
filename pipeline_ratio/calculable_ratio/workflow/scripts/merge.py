import pandas as pd
from collections import defaultdict

def get_path(file):
    steps = file.split("_")
    path = ' '.join((steps[-2], steps[-1]))
    return path.split('.csv')[0]

def stats(df):
    tot = len(df)
    TP = sum(df['TP'])/tot
    TN = sum(df['TN'])/tot
    FP, FN = 1-TP, 1-TN
    precision = TP + TN
    metrics = {'FP': FP, 'FN': FN, 'prec': precision}
    return metrics

def merge(files, metrics):
    
    #Metrics
    dico = defaultdict(list)
    
    for file in files:
        dico[get_path(file)].append(pd.read_csv(file))
    
    for key in dico:
        dico[key] = pd.concat(dico[key]).reset_index(drop=True)
    
    df_classif = pd.DataFrame({k: stats(dico[k]) for k in dico}).T
    df_classif.to_csv(metrics)

merge(snakemake.input["error"],
      snakemake.output["metrics"])

