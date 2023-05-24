import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn import metrics

def get_nn(file):
    steps = file.split("_")
    path = ''.join(steps[-1])
    return path.split('.npy')[0]

def roc(preds, annots, graph, matrix_out):
    
    #Metrics
    dico = defaultdict(list)
    
    for pred in sorted(preds):
        dico[get_nn(pred)].append(np.load(pred))
    
    for nn in dico:
        dico[nn] = np.concatenate(dico[nn])

    truth = np.concatenate([np.load(a) for a in sorted(annots)])
    truth = (truth[:, 0] > 0) & (truth[:, 1] > 0)
    truth = truth.astype(float)

    #Graph
    areas = {}
    opt_conf= {}

    for nn in dico:
        fpr, tpr, thresh = metrics.roc_curve(truth, dico[nn])
        areas[nn] = metrics.roc_auc_score(truth, dico[nn])
        plt.plot(fpr, tpr, 
                 label=f"{nn}, AUC={round(areas[nn], 3)}")
        opt = np.argmax(tpr - fpr)
        opt_preds = dico[nn] > thresh[opt]
        conf = metrics.confusion_matrix(truth, opt_preds)
        opt_conf[nn] = pd.DataFrame(conf)

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(graph)

    with open(matrix_out, 'w') as f:
        print(opt_conf, file=f)

roc(snakemake.input["preds"],
      snakemake.input["annot"],
      snakemake.output["roc"],
      snakemake.output["matrix"])



