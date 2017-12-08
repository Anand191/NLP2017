#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:43:18 2017

@author: anand
"""

from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import os

s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('blue', 4), ('red', 1)]
d = defaultdict(list)
for k, v in s:
    print(k,v)
    d[k].append(v)
    
#%%    
train_file = "ted/ted-train.txt"

def read(fname, max_lines=np.inf):
    """
    Reads in the data in fname and returns it as
    one long list of words. Also returns a vocabulary in
    the form of a word2index and index2word dictionary.
    """
    data = []
    # w2i will automatically keep a counter to asign to new words
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    start = "<s>"
    end = "</s>"
    
    with open(fname, "r") as fh:
        for k, line in enumerate(fh):
            if k > max_lines:
                break
            words = line.strip().split()
            # assign an index to each word
            for w in words:
                i2w[w2i[w]] = w # trick
            
            sent = [start] + words + [end]
            data.append(sent)

    return data, w2i, i2w
#%%

def train_ngram(data, N, k=0):
    """
    Trains an n-gram language model with optional add-k smoothing
    and additionaly returns the unigram model

    :param data: text-data as returned by read
    :param N: (N>1) the order of the ngram e.g. N=2 gives a bigram
    :param k: optional add-k smoothing
    :returns: ngram and unigram
    """
    ngram = defaultdict(Counter) # ngram[history][word] = #(history,word)
    unpacked_data = [word for sent in data for word in sent]
    unigram = defaultdict(float, Counter(unpacked_data)) # default prob is 0.0           
    
    master = []
    for sent in data:
        for i in range (len(sent)):
            history = sent[i]        
            word = []
            word.append(history)
            for j in range(1,N):
                word.append(sent[i+j])
            master.append(word)
            if (len(sent)-i == N):
                break

    return (ngram, unigram)

data, w2i, i2w = read(train_file)
#bigram, unigram = train_ngram(data, N=2, k=0)
# bigram_smoothed, unigram_smoothed = train_ngram(data, N=2, k=1)

#%%

