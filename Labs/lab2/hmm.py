#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:17:25 2017

@author: anand
"""

import numpy as np
from collections import defaultdict
from collections import namedtuple

#%%
test_data = """sleep cry laugh cry
cry cry laugh sleep"""

def test_reader(test_lines):
    for line in test_lines.splitlines():
        yield line.split()

test_set = list(test_reader(test_data))

# read in train data
train_data = """laugh/happy cry/bored cry/hungry sleep/happy
cry/bored laugh/happy cry/happy sleep/bored
cry/hungry cry/bored sleep/happy"""

# for convenience, we define a Observation-State pair class
Pair = namedtuple("Pair", ["obs", "state"])
Pair.__repr__ = lambda x: x.obs + "/" + x.state

def train_reader(train_lines):
    for line in train_data.splitlines():
        # a pair is a string "observation/state" so we need to split on the "/"
        yield [Pair(*pair.split("/")) for pair in line.split()]

training_set = list(train_reader(train_data))

# print the results
print("test set (observations):")
for seq in test_set:
    print(seq)
print("\ntraining set (observation/state pairs):")
for seq in training_set:
    print(seq)

#%%
# create mappings from state/obs to an ID
state2i = defaultdict(lambda: len(state2i))
obs2i = defaultdict(lambda: len(obs2i))

for seq in training_set:
    for example in seq:
        state_id = state2i[example.state]
        obs_id = obs2i[example.obs]
        
print("\nOur vocabularies:")
print(state2i)
print(obs2i)

#%%
# we can get the number of states and observations from our dictionaries
num_states = len(state2i)
num_observations = len(obs2i)

# this creates a vector of length `num_states` filled with zeros
counts_start = np.zeros(num_states)

# now we count 1 every time a sequence starts with a certain state
# we look up the index for the state that we want to count using the `state2i` dictionary
for seq in training_set:
    counts_start[state2i[seq[0].state]] += 1.

print(counts_start)

#%%
# since p_start is a numpy object, we can call sum on it; easy!
total = counts_start.sum()

# normalize: divide each count by the total
p_start = counts_start / total  
print('start', '-->', p_start)

#%%
# we can transition from any state to any other state in principle,
# so we create a matrix filled with zeros so that we can count any pair of states
# in practice, some transitions might not occur in the training data
counts_trans = np.zeros([num_states, num_states])

# for the final/stop probabilities, we only need to store `num_states` values.
# so we use a vector
counts_stop = np.zeros(num_states)

# now we count transitions, one sequence at a time
for seq in training_set:
    for i in range(1, len(seq)):
        
        # convert the states to indexes
        prev_state = state2i[seq[i-1].state]
        current_state = state2i[seq[i].state]
        
        # count
        counts_trans[current_state, prev_state] += 1.

# count final states
for seq in training_set:
    state = state2i[seq[-1].state]
    counts_stop[state] += 1.

# print the counts
print("Transition counts:")
print(counts_trans)

print("Final counts:")
print(counts_stop)
print(counts_trans.sum(0))
#%%
total_per_state = counts_trans.sum(0) + counts_stop
print("Total counts per state:\n", total_per_state, "\n")

# now we normalize
# here '/' works one column at a time in the matrix
p_trans = counts_trans / total_per_state
print("Transition probabilities:\n", p_trans)

# here '/' divides the values in each corresponding index in the 2 vectors
p_stop = counts_stop / total_per_state
print("Final probabilities:\n", p_stop, "\n")

#%%
# now we create a matrix to keep track of emission counts
# in principle any states can emit any observation
# so we need a matrix again
counts_emiss = np.zeros([num_observations, num_states])

# count
for seq in training_set:
    for obs, state in seq:
        obs = obs2i[obs]
        state = state2i[state]
        counts_emiss[obs][state] += 1.

# normalize
p_emiss = counts_emiss / counts_emiss.sum(0)

print("emission counts:\n", counts_emiss)
print("p_emiss:\n", p_emiss)

#%%
def almost_one(p, eps=1e-3):
    return (1.-eps) < p < (1. + eps)

def sanity_check(p_start=None, p_trans=None, p_stop=None, p_emiss=None):
    p_em = np.sum(p_emiss,axis=0)
    print (p_em)
    p_tot = np.sum(np.vstack((p_trans,p_stop)),axis=0)
    print (p_tot)
    assert almost_one(np.sum(p_start))
    for i in range(len(p_em)):
        assert almost_one(p_em[i])
        assert almost_one(p_tot[i])
        
try:
    sanity_check(p_start=p_start, p_trans=p_trans, p_stop=p_stop, p_emiss=p_emiss)
    print("All good!")
except AssertionError as e:
    print("There was a problem: %s" % str(e))