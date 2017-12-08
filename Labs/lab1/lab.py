#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:35:52 2017

@author: anand
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
from itertools import combinations
#%% Tokens and Vocabulary
dir1 = os.getcwd()

with open(os.path.join(dir1,"wsj/sec02-21.gold.tagged")) as f:
    wd = [word for line in f for word in line.split()]
    words = [wd[i].split('|')[0] for i in range (len(wd))]
    print (len(words))
    print (len(np.unique(words)))
    
    c = Counter(words)
    cc= Counter(wd)
    print (len(c))
    
#%% Lower case Vocabulary
words2 = []
for w in words:
    words2.append(w.lower())
c2 = Counter(words2)
print(len(c2))

#%% Zipf's law
dd = c.most_common()
d = cc.most_common()
rank,freq = [],[]
for i in range(0,len(dd)):
    rank.append(i)
    freq.append(dd[i][1])
plt.loglog(rank,freq)
plt.show()

#%% 20 most common words
labels = [dd[i][0] for i in range(20)]
data = np.column_stack((labels,freq[0:20]))
df = pd.DataFrame(data,columns=['Words','Frequencies'])
print(df)


#%% Different POS
names = [d[i][0].split('|')[0] for i in range(len(d))]
pos = [d[i][0].split('|')[1] for i in range(len(d))]
group = np.column_stack((pos,names))
cpos = Counter(pos)
mpos = cpos.most_common()
print(mpos[0:10])
tags = [mpos[i][0] for i in range(10)]

#%% POS with words from that class
mcw = []
for t in tags:
    rows = np.where(group[:,0]==t)[0]
    temp = Counter(group[rows,1])
    temp2 = temp.most_common()
    wds = [temp2[i][0] for i in range(3)]
    mcw.append(wds)

out = np.column_stack((tags,mcw))
df2 = pd.DataFrame(out,columns=['POS','Word1','Word2','Word3'])
print(df2)

#%%ambiguous as % of vocab
cnames = Counter(names)
mnames = cnames.most_common()
vocab = [mnames[i][0] for i in range(len(mnames))]
maxl = max([mnames[i][1] for i in range(len(mnames))])
ambiguous = []
for i in range(len(mnames)):
    if (mnames[i][1] >1):
        ambiguous.append(mnames[i][0])
    if (mnames[i][1]==maxl):
        print(mnames[i][0])
ambi = len(ambiguous)
print(ambi/len(names))
#%% ambiguous POS
mct = []
for v in vocab:
    rows = np.where(group[:,1]==v)[0]
    temp = Counter(group[rows,0])
    temp2 = temp.most_common()
    #print (temp2)
    wds = [temp2[i][0] for i in range(len(temp2))]
    mct.append(wds)
#%%10 most ambiguous POS combo
pairs = []
for i in range(len(mct)):
    pairs.append(list(combinations(mct[i],2)))
flat_list = [item for sublist in pairs for item in sublist]
p1 = Counter(tuple(sorted(tup)) for tup in flat_list)
#p = Counter(flat_list)
print(p1.most_common(10))    

#%% print most ambiguous
for j in range(len(vocab)):
    if (len(mct[j])==maxl):
        print ("{}:{}".format(vocab[j],mct[j]))
        
#%% % of dataset ambiguous
cd = dict(c)
s = 0
for each in cd:
    if np.any(np.asarray(ambiguous) == each.split('|')[0]):
        s += cd[each]
print ("{}% of dataset is ambiguous".format((s*100.)/len(words)))

#%%
testf = "wsj/sec00.gold.tagged"
with open(testf,'r') as f:
    td = [word for line in f for word in line.split()]
    words2 = [td[i].split('|')[0] for i in range (len(td))]
    pos2 = [td[i].split('|')[1] for i in range (len(td))]
    dst = np.column_stack((words2,pos2))
    dfn = pd.DataFrame(dst,columns=['Words','POS'])
    print ("No. of token =",len(words2))
    #print (len(np.unique(words)))
    
    c = Counter(words2)
    cc= Counter(td)
    
#%%
words3 = np.unique(words2)
unseen = []
for w in words3:
    if(w not in vocab):
        unseen. append(w)
#%%
max_pos = []
for u in unseen:
    rows = np.where(u==dst[:,0])[0]
    max_pos.append(dst[rows,1])
    
flattened = [item for sublist in max_pos for item in sublist]
        



    

    
