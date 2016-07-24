# -*- coding: utf-8 -*-

from __future__ import print_function
import function
import numpy as np

def nls_bp(A, B, init = None):
    '''
    this function is for 2 subproblems of
    Least Squares Error.
    this function itself require 2 sub functions:
        normal_eq_comb & column_group, explained respectively f
        from function.py
    '''

    #NOTE: min_x ||AX - B||, where all elements of X is non-negative.

    """ Parameters
    - A
        shape is (m, n).

    - B
        shape is (m, k).
    both of teme should be numpy.ndarray

    - init
        shape is (n, k).
        this paramete is source of output of this function: "X".
        If init is None, initialize X 0-matrix.
    """

    """ Outputs
    - X
        shape is (n, k).
        X =  argmin_{X ≥ 0} ||AX - B||_{F}^{2}
    """

    """
    In this calculation, A^{T}A and A^{T}B will be used so frequently.
    So, calculate them and define them as 2 constants, AtA & AtB.
    AtA.shape = (n, n)
    AtB.shape = (n, k)
    """
    AtA = np.dot(A.T, A)
    AtB = np.dot(A.T, B)
    (n, k) = AtB.shape
    ITER = n * 5 # one threshold of iteration

    if init is None:
        # initialize X as 0-matrix
        X = np.zeros((n, k))
        Y = - AtB
        PassSet = np.zeros((n, k), dtype = bool)
        #NOTE:
        #PassSet is alternative of index set of V in reference.
        num_cholesky = 0
        #`num_cholesky` is the times of Cholesky decomposition.
        num_eq = 0
        #`num_eq` is the number of equations.
    else:
        #init is defined by user.
        #execute first iteration.
        #that is, calculate Eq(3-8) and (3-9) in reference.
        PassSet = init > 0
        X, num_cholesky, num_eq = function.normal_eq_comb(AtA, AtB, PassSet)
        Y = np.dot(AtA, X) - AtB

    '''
    Here, I define vector alpha and beta. Both of them shape = k
    '''
    tmp_var = 3
    alpha = np.zeros(k)
    alpha[:] = tmp_var
    beta = np.zeros(k)
    beta[:] = n + 1

    '''
    Here, I define the index set of F, G and V.
    indx set: {0, 1, ..., n}
    columns: {0, 1, ..., k}

    remember that when init is None,
    F = ø and G = {0, 1, ..., n}
    Also, in ths function, if init is None, all PassSet' elements is "FALSE".
    Collect index 'i' where x_i < 0 or y_i < 0.
    '''
    nopt_set = np.logical_and(Y < 0, ~PassSet)
    #shape: (n, k)
    infeasible_set = np.logical_and(X < 0, PassSet)
    #shape: (n, k)

    ngood = np.sum(nopt_set, axis = 0) + np.sum(infeasible_set, axis = 0)
    nopt_colset = ngood > 0
    nopt_cols = np.nonzero(nopt_colset)[0]

    n_iter = 0
    num_backup = 0
    #backup is provoked when full-exchange doesn't work.

    while nopt_cols.size > 0:
        n_iter += 1

        if ITER > 0 and n_iter > ITER:
            break

        cols_set1 = np.logical_and(nopt_colset, ngood < beta)

        tmp1 = np.logical_and(nopt_colset, ngood >= beta)
        tmp2 = alpha >= 1

        cols_set2 = np.logical_and(tmp1, tmp2)
        cols_set3 = np.logical_and(tmp1, ~tmp2)

        cols1 = np.nonzero(cols_set1)[0]
        cols2 = np.nonzero(cols_set2)[0]
        cols3 = np.nonzero(cols_set3)[0]

        if len(cols1) > 0:
            alpha[cols1] = tmp_var
            beta[cols1] = ngood[cols1]
            true_set = np.logical_and(nopt_set, np.tile(cols_set1, (n, 1)))
            false_set = np.logical_and(infeasible_set, np.tile(cols_set1, (n, 1)))
            PassSet[true_set] = True
            PassSet[false_set] = False

        if len(cols2) > 0:
            alpha[cols2] = alpha[cols2] - 1
            tmp_tile = np.tile(cols_set2, (n, 1))
            true_set = np.logical_and(nopt_set, tmp_tile)
            false_set = np.logical_and(infeasible_set, tmp_tile)
            PassSet[true_set] = True
            PassSet[false_set] = False

        if len(cols3) > 0:
            for col in cols3:
                c_set = np.logical_or(nopt_set[:, col], infeasible_set[:, col])
                change = np.max(np.nonzero(c_set)[0])
                PassSet[change, col] = ~PassSet[change, col]
                num_backup += 1

        (X[:, nopt_cols], tmp_cholesky, tmp_eq) = function.normal_eq_comb(AtA, AtB[:, nopt_cols], PassSet[:, nopt_cols])
        num_cholesky += tmp_cholesky
        num_eq += tmp_eq
        X[abs(X) < 1e-12] = 0
        Y[:, nopt_cols] = np.dot(AtA, X[:, nopt_cols]) - AtB[:, nopt_cols]
        Y[abs(Y) < 1e-12] = 0

        nopt_mask = np.tile(nopt_colset, (n, 1))
        nopt_set = np.logical_and(np.logical_and(nopt_mask, Y < 0), ~PassSet)
        infeasible_set = np.logical_and(np.logical_and(nopt_mask, X < 0), PassSet)
        ngood = np.sum(nopt_set, axis = 0) + np.sum(infeasible_set, axis = 0)
        nopt_colset = ngood > 0
        nopt_cols = np.nonzero(nopt_colset)[0]

    return X , (Y, num_cholesky, num_eq, num_backup)
