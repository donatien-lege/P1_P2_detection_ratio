import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.stats as st

def get_path(file):
    steps = file.split("_")
    path = ' '.join((steps[-2], steps[-1]))
    return path.split('.csv')[0]

def col(x, dico):
    return plt.get_cmap("nipy_spectral")(x/len(dico))

def format_ax(ax, title, xlabel=''):
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probabilité")

def conf(a):
    mean = np.mean(a)
    try:
        inf, sup = st.t.interval(0.95, len(a)-1, 
                                loc=mean, 
                                scale=st.sem(a))
        return round(mean, 4), round(sup-mean, 4)
    except ZeroDivisionError:
        return round(mean, 4), np.nan

def plot_error(dico, key_1, key_2):
    errors = {}
    for df in dico:
        P1, P2 = np.abs(dico[df][key_1]), np.abs(dico[df][key_2])
        errors[df] = *conf(P1), *conf(P2)
    errors = pd.DataFrame(errors).T
    errors.columns = (key_1, f"err_{key_1}", key_2, f"err_{key_2}")
    return errors
    
def plot_ratio(dico, key):
    errors = {}
    for df in dico:
        r_12 = np.abs(dico[df][key])
        errors[df] = conf(r_12)
    
    errors = pd.DataFrame(errors).T
    errors.columns = ("MAE_ratio", "err_ratio")
    return errors

    
def stats(df):
    try:
        TP = 100*(sum(df['RP'] & df['RA']))/sum(df['RA'])
        TN = 100*(sum(~df['RP'] & ~df["NAN"] & ~df['RA']))/sum(~df['RA'])
        FP, FN = round(100-TP, 2), round(100-TN, 2)
        precision = 100*(sum(df['RP'] == df['RA']) - sum(df["NAN"]))/len(df)
        metrics = {'FP': FP, 'FN': FN, 'prec': round(precision, 2)}
    except ZeroDivisionError:
        metrics = {'FP': 0, 'FN': 0, 'prec': 0}
    return metrics


def merge(files, clasf, metrics):
    
    dico = defaultdict(list)
    
    for file in files:
        dico[get_path(file)].append(pd.read_csv(file))
    
    for key in dico:
        dico[key] = pd.concat(dico[key]).reset_index(drop=True)


    #Erreurs verticales
    hz = plot_error(dico, 
               key_1='hz_P1',
               key_2='hz_P2')
    
    #Erreurs verticales
    vt = plot_error(dico, 
               key_1='vt_P1',
               key_2='vt_P2')

    #Erreurs ratio
    ratio = plot_ratio(dico, 
               key='Ratio')
    
    df_classif = pd.DataFrame({k: stats(dico[k]) for k in dico}).T
    plt.figure(figsize=(10, 10))
    df_classif.plot.bar(rot=0)
    plt.xticks(rotation = 45)
    plt.savefig(clasf, bbox_inches='tight')
    metr = pd.concat([hz, vt, ratio, df_classif], axis=1)
    print(metr)
    metr.to_csv(metrics)


merge(snakemake.input["error"],
      snakemake.output["clasf"],
      snakemake.output["metrics"])

