# -*- coding: utf-8 -*-
# Python 2.7.11 :: Anaconda

from __future__ import print_function
import numpy as np


num_of_reviewer = 943
num_of_item = 1682

def makeMovieDictionary(path = './ml-100k/'):
    '''
    this function returns dictionary of movie.
    Key is number of movie.
    Value is title of movie.
    '''
    movies = {}

    for line in open(path + 'u.item', 'r'):
        (number, title) = line.split('|')[0:2]
        movies[number] = title

    return movies

def loadMovie(filename = None, path = './ml-100k/'):
    '''
    this function returns dictionary of ratings of movies by users.
    Key is user_number(str) and Value.
    Value contains dictionary whose Key is movies' title and Value is rating.
    '''
    movies = makeMovieDictionary()

    if filename == None:
        filename = 'u.data'
    prefs = {}

    for line in open(path + filename, 'r'):
        (user, movie_id, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movie_id]] = float(rating)

    return prefs

def convertDicToArray(prefs = None, cols = num_of_reviewer, rows = num_of_item):
    '''
    loadMovie関数で作った辞書をnumpy ndarrayに変換する
    '''
    R = np.zeros((cols, rows), dtype = np.float32)
    movies = makeMovieDictionary()

    if prefs == None:
        prefs = loadMovie()

    for person in prefs:
        for item in prefs[person]:
            item_index = int(movies.keys()[movies.values().index(item)]) - 1
            R[int(person) - 1, item_index] = prefs[person][item]

    return R
