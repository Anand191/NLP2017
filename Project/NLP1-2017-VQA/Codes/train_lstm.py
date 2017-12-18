#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:48:55 2017

@author: anand
"""

import gzip
import json
import numpy as np
import h5py
import random
import time
from collections import defaultdict
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import sys
import copy
import pickle

from lstm import LSTM

torch.manual_seed(1)
random.seed(1)


CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)
ACTIVATIONS = {'relu':F.relu,'sigmoid':F.sigmoid,'tanh':F.tanh,'softmax':F.softmax}

#%% Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["words", "tag", "img_feat"])

def read_dataset(questions_path, annotations_path, image_features_path, img_features2id_path, imgid2imginfo_path):

    with open(imgid2imginfo_path, 'r') as file:
        imgid2info = json.load(file)

    # load image features from hdf5 file and convert it to numpy array
    img_features = np.asarray(h5py.File(image_features_path, 'r')['img_features'])

    # load mapping file
    with open(img_features2id_path, 'r') as f:
        visual_feat_mapping = json.load(f)['VQA_imgid2id']

    with gzip.GzipFile(questions_path, 'r') as file:
        for line in file:
             questions = eval(line.decode())
        #questions = question['questions']
        #questions = json.loads(file.read())

    with gzip.GzipFile(annotations_path, 'r') as file:
        for line in file:
             annotations = eval(line.decode())
        #annotations = annotation['annotations']
        #annotations = json.loads(file.read())

    for line in range(len(questions['questions'])):
        words = questions['questions'][line]['question'].lower().strip()
        tag = annotations['annotations'][line]['multiple_choice_answer']
        img_id = questions['questions'][line]['image_id']
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        yield Example(words=[w2i[x] for x in words.split(" ")],
                      tag=t2i[tag],
                      img_feat=img_feat)


# Read in the data
train = list(read_dataset( "../data/vqa_questions_train.gzip",
                           "../data/vqa_annotatons_train.gzip",
                           "../data/features/VQA_image_features.h5",
                           "../data/features/VQA_img_features2id.json",
                            "../data/imgid2imginfo.json"))

w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/vqa_questions_valid.gzip",
                        "../data/vqa_annotatons_valid.gzip",
                        "../data/features/VQA_image_features.h5",
                        "../data/features/VQA_img_features2id.json",
                        "../data/imgid2imginfo.json"))
                        
nwords = len(w2i)
ntags = len(t2i)
#%%
model = LSTM(nwords,
                 300,
                 2048,
                 ntags,
                 256,
                 1,
                 [512],
                 ACTIVATIONS['relu'])
if CUDA:
    model.cuda()

print(model)

#%%Preprocess raw data
def minibatch(data, batch_size=256):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]        

def preprocess(batch):
    """ Add zero-padding to a batch. """

    tags = [example.tag for example in batch]

    # add zero-padding to make all sequences equally long
    seqs = [example.words for example in batch]
    max_length = max(map(len, seqs))
    seqs = [seq + [PAD] * (max_length - len(seq)) for seq in seqs]
    img = [example.img_feat.tolist() for example in batch]
    return seqs, tags, img


def get_variable(x):
    """Get a Variable given indices x"""
    tensor = torch.cuda.LongTensor(x) if CUDA else torch.LongTensor(x)
    return Variable(tensor)
def get_image(x):
    tensor = torch.cuda.FloatTensor(x) if CUDA else torch.FloatTensor(x)
    return Variable(tensor)

def evaluate(model, data):
    """Evaluate a model on a data set."""
    correct = 0.0

    for batch in minibatch(data):

        seqs, tags, image = preprocess(batch)
        scores = model(get_variable(seqs), get_image(image))
        _, predictions = torch.max(scores.data, 1)
        targets = torch.cuda.LongTensor(tags) if CUDA else torch.LongTensor(tags)
        #targets = get_variable(tags)
        #correct += torch.eq(predictions, targets).sum().data[0] 
        correct += torch.eq(predictions, targets).sum()

    return correct, len(data), correct/len(data)

#%%Model and Train Step
##Shift Train Step to model def
tl,ta,va = [],[],[]
st,sa = [],[]
optimizer = optim.Adam(model.parameters(),0.001)
#%%
EPOCHS = 30
for ITER in range(EPOCHS):

    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    updates = 0

    for batch in minibatch(train, 256):

        updates += 1

        # pad data with zeros
        seqs, tags, image = preprocess(batch)

        # forward pass
        scores = model(get_variable(seqs), get_image(image))
        targets = get_variable(tags)
        loss = nn.CrossEntropyLoss()
        output = loss(scores, targets)
        train_loss += output.data[0]

        # backward pass
        model.zero_grad()
        output.backward()

        # update weights
        optimizer.step()

    tl.append(train_loss/updates)
    st.append(ITER)
    print("iter %r/%r: avg train loss=%.4f, time=%.2fs" %
          (ITER,EPOCHS, train_loss/updates, time.time()-start))

    # evaluate
    if((ITER+1)%3==0):
        _, _, acc_train = evaluate(model, train)
        _, _, acc_dev = evaluate(model, dev)
        ta.append(acc_train)
        va.append(acc_dev)
        sa.append(ITER)
        print("iter %r/%r: train acc=%.4f  dev acc=%.4f" % (ITER,EPOCHS, acc_train, acc_dev))
#%%
saved_model=copy.deepcopy(model)
with open(r"my_model2.pkl","wb") as f:
	pickle.dump(model,f)
#%%
with open("results.txt",'w') as f:
    f.write(str(tl))
    f.write("BK")
    f.write(str(ta))
    f.write("BK")
    f.write(str(va))
    f.write("BK")
    f.write(str(st))
    f.write("BK")
    f.write(str(sa))