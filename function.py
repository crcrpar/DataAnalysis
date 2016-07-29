# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import numpy.linalg as nla

def normal_eq_comb(AtA, AtB, PassSet = None):
    num_cholesky = 0
    num_eq = 0
    if AtB.size == 0:
        Z = np.zeros([])

    elif (PassSet is None) or np.all(PassSet):
        Z = nla.solve(AtA, AtB)
        num_cholesky = 1
        num_eq = AtB.shape[1]

    else:
        Z = np.zeros(AtB.shape) #(n, k)
        if PassSet.shape[1] == 1:
            if np.any(PassSet):
                cols = np.nonzero(PassSet)[0]
                Z[cols] = nla.solve(AtA[np.ix_(cols, cols)], AtB[cols])
                num_cholesky = 1
                num_eq = 1
        else:
            groups = column_group(PassSet)

            for g in groups:
                cols = np.nonzero(PassSet[:, g[0]])[0]

                if cols.size > 0:
                    ix1 = np.ix_(cols, g)
                    ix2 = np.ix_(cols, cols)

                    Z[ix1] = nla.solve(AtA[ix2], AtB[ix1])
                    num_cholesky += 1
                    num_eq += len(g)
                    num_eq += len(g)
    return Z, num_cholesky, num_eq

def column_group(B):
    #NOTE:
    # B's elements are bool.
    init = [np.arange(0, B.shape[1])]
    pre = init
    after = []

    for i in range(0, B.shape[0]):
        all_ones = True
        vec = B[i]
        
        for cols in pre:
            if len(cols) == 1:
                after.append(cols)
            else:
                all_ones = False
                sub_vec = vec[cols]
                trues = np.nonzero(sub_vec)[0]
                falses = np.nonzero(~sub_vec)[0]

                if trues.size > 0:
                    after.append(cols[trues])
                if falses.size > 0:
                    after.append(cols[falses])
        pre = after
        after = []
        if all_ones:
            break
    return pre
