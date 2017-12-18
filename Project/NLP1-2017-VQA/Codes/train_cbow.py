# coding: utf-8

"""
Deep CBOW (with minibatching)

Based on Graham Neubig's DyNet code examples:
  https://github.com/neubig/nn4nlp2017-code
  http://phontron.com/class/nn4nlp2017/

"""
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

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
#import requests
import copy
import pickle

from cbow import CBOW

#%%
torch.manual_seed(1)
random.seed(1)


CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)

ACTIVATIONS = {'relu':F.relu,'sigmoid':F.sigmoid,'tanh':F.tanh,'softmax':F.softmax}

#%%
# Functions to read in the corpus
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

    with gzip.GzipFile(annotations_path, 'r') as file:
        for line in file:
             annotations = eval(line.decode())

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
itr2t = {}
for key,value in t2i.items():
    itr2t[value] = key
    
dev = list(read_dataset("../data/vqa_questions_valid.gzip",
                        "../data/vqa_annotatons_valid.gzip",
                        "../data/features/VQA_image_features.h5",
                        "../data/features/VQA_img_features2id.json",
                        "../data/imgid2imginfo.json"))

nwords = len(w2i)
ntags = len(t2i)
#%% Create Graph
def len_data_dist(data):
    l = []
    for i in range(len(train)):
        l.append(len(train[i].words))
    return int(np.percentile(l,90))

percentile = len_data_dist(train)

model = CBOW(nwords,
                 300,
                 2048,
                 ntags,
                 [512],
                 ACTIVATIONS['relu'],
                 percentile)
if CUDA:
    model.cuda()

print(model)

#%% Support functions
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

        correct += torch.eq(predictions, targets).sum()

    return correct, len(data), correct/len(data)

# =============================================================================
# def predict(model, data):
#     """Evaluate a model on a data set."""
#     #correct = 0.0
# 
# 
#     seqs, tags, image = preprocess(data)
#     scores = model(get_variable(seqs), get_image(image))
#     _, predictions = torch.max(scores.data, 1)
#         #targets = torch.cuda.LongTensor(tags) if CUDA else torch.LongTensor(tags)
# 
#         #correct += torch.eq(predictions, targets).sum()
# 
#     return seqs, tags, predictions
# =============================================================================

#%%
tl,ta,va = [],[],[]
st,sa = [],[]
optimizer = optim.Adam(model.parameters(),0.001)
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
# =============================================================================
# plt.figure(figsize=(12,12))
# plt.subplot(131)
# plt.plot(st,tl)
# plt.xlabel("epochs")
# plt.ylabel("training loss")
# plt.subplot(132)
# plt.plot(sa,va)
# plt.xlabel("epochs")
# plt.ylabel("validation accuracy")
# plt.subplot(133)
# plt.plot(sa,ta)
# plt.xlabel("epochs")
# plt.ylabel("training accuracy")
# plt.tight_layout()
# plt.savefig("Learning_Curves.png")
# print("\n")
# =============================================================================

#%%
        
        
