import numpy as np

def gen_labels(in_annot, 
               out_annot):

    peaks = np.load(in_annot).astype(int)

    p1, p2 = peaks[:, 0], peaks[:, 1]
    mask = (p1 > 0) & (p2 > 0)
    annots = mask.astype(float)
    
    np.save(out_annot, annots)

gen_labels(snakemake.input['annots'],
snakemake.output['annots'])
    

    
