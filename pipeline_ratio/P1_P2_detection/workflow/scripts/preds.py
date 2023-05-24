import numpy as np
from scipy.signal import argrelmax, find_peaks
import inspect as isp

def min_len(func):
    def wrapper(*args):
        pred = func(*args)
        if len(pred) <= 1:
            return np.array([0, 0])
        return pred
    return wrapper

class Combination:
    
    @min_len
    def curve(self, pdf, curve):
        maxima = argrelmax(curve, order=5)[0]
        return np.sort(maxima)[:2]
    
    @min_len
    def nothing(self, pdf, curve):
        maxima = find_peaks(pdf)[0]
        maxima = sorted(maxima, key=lambda x: pdf[x])[-2:]
        return np.sort(maxima)

    @min_len
    def both(self, pdf, curve):
        maxima = argrelmax(curve)[0]
        maxima = sorted(maxima, key=lambda x: pdf[x])[-2:]
        return np.sort(maxima)

        
if __name__ == '__main__':

    pdf = np.load(snakemake.input['pdf'], allow_pickle=True)
    curvature = np.load(snakemake.input['curvature'], allow_pickle=True)
    instance = Combination()
    methods = dict(isp.getmembers(instance, predicate=isp.ismethod))
    func = methods[snakemake.wildcards['c']]
    candidates = [func(p, c) for p, c in zip(pdf, curvature)]
    
    np.save(snakemake.output[0], np.stack(candidates))