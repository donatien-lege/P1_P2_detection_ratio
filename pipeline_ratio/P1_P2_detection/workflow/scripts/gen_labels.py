import numpy as np


def gen_labels(in_annot, 
               out_annot, 
               in_pulse, 
               out_pulse, 
               size):

    peaks = np.load(in_annot).astype(int)
    pulses = np.load(in_pulse)

    idx = range(len(peaks))
    p1, p2 = peaks[:, 0], peaks[:, 1]
    div = pulses[idx, p1] / pulses[idx, p2]
    mask = (div > 0.1) & (div < 10)
    peaks = peaks[mask] 
    pulses = pulses[mask] 
    
    annot = []
    for row in peaks:
        arr = np.arange(size)     
        s = np.zeros(size)
        for peak in row:
            peak1 = np.exp((-0.5*(peak-arr)**2))
            peak1 *= np.exp(0.5)
            s += peak1
        annot.append(s)
    annot = np.stack(annot)
    np.save(out_annot, annot)

    pulses = np.save(out_pulse, pulses)

gen_labels(snakemake.input['annots'],
snakemake.output['annots'],
snakemake.input['pulses'],
snakemake.output['pulses'],
snakemake.config['size'])
    

    
