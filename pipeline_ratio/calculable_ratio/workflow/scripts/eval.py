import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def eval(pred_path, annot_path, out_path, out_matrix, threshold):
    
    #chargements
    preds = np.load(pred_path, allow_pickle=True)
    peaks = np.load(annot_path, allow_pickle=True).astype(bool)
    df_error = pd.DataFrame()
    
    p1, p2 = peaks[:, 0], peaks[:, 1]
    mask = (p1 > 0) & (p2 > 0)
    annots = mask.astype(bool)

    preds = preds > threshold
    df_error["TP"] = preds & annots
    df_error["TN"] = ~preds & ~annots
    df_error.to_csv(out_path, index=False)
    matrix = confusion_matrix(annots, preds)
    if matrix.shape == (1, 1):
        matrix = np.array([[int(matrix), 0], [0, 0]])
        if sum(annots):
            matrix = matrix.T
        
    np.save(out_matrix, matrix)

    
eval(snakemake.input['pred'], 
    snakemake.input['annot'], 
    snakemake.output['errors'],
    snakemake.output['matrices'],
    snakemake.config['threshold'])