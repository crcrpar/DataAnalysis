# -*- coding: utf-8
# Python 2.7.11

from __future__ import print_function
import argparse
import data
import function
import math
import numpy as np
import ace
import time
import pandas as pd

def data_load(filename = None, path = './ml-100k/'):
    cols = data.num_of_reviewer
    rows = data.num_of_item

    pref = data.loadMovie(filename = filename)
    R = data.convertDicToArray(prefs = pref, cols = cols, rows = rows)

    return R

def fast_nmf(R, k = 20, m_iter = 100, eps = 0.01, alpha = 1, beta = 1):
    (m, n) = R.shape
    P_ = np.random.rand(m, k)
    Q_ = np.random.rand(k, n)

    A_p = np.concatenate((Q_.T, np.multiply(math.sqrt(2 * alpha), np.identity(k))))
    print(A_p.shape)
    #A_p.shape = (n+k, k)
    B_p = np.concatenate((R.T, np.zeros((k,m))))
    print(B_p.shape)
    #B_p.shape = (n+k, m)
    A_q = np.concatenate((P_, np.multiply(math.sqrt(2 * beta), np.identity(k))))
    print(A_q.shape)
    #A_q.shape = (m+k, k)
    B_q = np.concatenate((R, np.zeros((k, n))))
    print(B_q.shape)
    #B_q.shape = (m+k, n)
    print("begin tuning...\n")

    for i in range(m_iter):
        start = time.time()
        sol = ace.nls_bp(A_p, B_p, init=P_.T)
        P_ans = sol[0].T
        sol = ace.nls_bp(A_q, B_q, init=Q_)
        Q_ans = sol[0]
        elapsed = time.time() - start
        print("#epoch:{}\ntime:{}".format(i, elapsed))
    return P_ans, Q_ans

def predictor(R, P, Q, out, filename, path = './ml-100k/'):
    R_hat = np.dot(P, Q)
    result = pd.read_table(path + filename, header = None)
    user_id = result.ix[:,0] -1
    movie_id = result.ix[:,1] - 1
    rate = result.ix[:,2]

    predict = np.asarray([R_hat[user_id[i],movie_id[i]] for i in user_id ])

    d_f = pd.DataFrame({"user_id": user_id, "movie_id": movie_id, "answer_rate":rate, "predicted":predict})

    h_file = filename.split('.')[0]
    csv_file = h_file + '.csv'
    if out >= 0:
        d_f.to_csv(csv_file)
        print("#save the result of test into csv file: {}".format(csv_file))

    return d_f


def main():
    parser = argparse.ArgumentParser(description='Define source file, the # of latent features and learning loop and whether output csv file of result.')
    parser.add_argument('--source', '-s', default = 'u1.base', help = 'u{i}.base i in {1,...,5}  u.data, ua.base or ub.base, --s')
    parser.add_argument('--latent_features', '-l', default = 20, type = int, help = 'define the number of latent features. --l')
    parser.add_argument('--learning_loop', '-n', default = 50, type = int, help = 'Define max iteration. --n')
    parser.add_argument('--csv_file', '-c', default = 0, type = int, help = 'If you do not want result_csv file, negative integer')
    args = parser.parse_args()

    f_name = args.source
    k_args = args.latent_features
    max_iter = args.learning_loop
    csv_ = args.csv_file

    R = data_load(filename = f_name)
    print("begin learning...")

    if f_name != 'u.data':
        test_file = f_name.split('.')[0] + '.test'

        start = time.time()
        P, Q = fast_nmf(R = R, k = k_args, m_iter = max_iter)
        elapsed = time.time() - start

        print("#DONE!")
        print("P.shape:{}, Q.shape:{}".format(P.shape, Q.shape))
        print("Time spent on factorization: {}".format(elapsed))

        d_f = predictor(R, P, Q, out = csv_, filename = test_file)

    else:
        start = time.time()
        P, Q = fast_nmf(R = R, k = k_args, m_iter = max_iter)
        elapsed = time.time() - start

        print("#DONE!")
        print("P.shape:{}, Q.shape:{}".format(P.shape, Q.shape))
        print("Time spent on factorization: {}".format(elapsed))



if __name__ == '__main__':
    main()
