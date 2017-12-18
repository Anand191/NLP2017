#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 14:30:31 2017

@author: anand
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
import requests
import copy
import pickle

from cbow import CBOW

CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)

#%%
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
PAD = w2i["<pad>"]

# One data point
Example = namedtuple("Example", ["words", "tag", "img_feat","img_id"])

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
                      img_feat=img_feat,
                      img_id = img_id)
        
        
# Read in the data
train = list(read_dataset( "../data/vqa_questions_train.gzip",
                           "../data/vqa_annotatons_train.gzip",
                           "../data/features/VQA_image_features.h5",
                           "../data/features/VQA_img_features2id.json",
                            "../data/imgid2imginfo.json"))

w2i = defaultdict(lambda: UNK, w2i)
itr2t,itr2w = {},{}
for key,value in w2i.items():
    itr2w[value] = key
    
for key,value in t2i.items():
    itr2t[value] = key
    
#%%
with open("../data/imgid2imginfo.json", 'r') as file:
    imgid2info = json.load(file)
    t_idx = np.random.choice(len(train),10)
    train_pred = [train[i] for i in t_idx]
    idx = [train[i].img_id for i in t_idx]
    j=0
    for i in idx:
        url = imgid2info[str(i)]['flickr_url']
        response = requests.get(url)
        if response.status_code == 200:
            with open("./train{}.jpg".format(j), 'wb') as f:
                f.write(response.content)
        j +=1
#%% Functions to create test set and get images for prediction
wt2i = defaultdict(lambda: len(wt2i))
tt2i = defaultdict(lambda: len(tt2i))
Example2 = namedtuple("Example", ["words", "tag", "img_feat", "img_id"])
def read_test(questions_path, annotations_path, image_features_path, img_features2id_path, imgid2imginfo_path):

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
    l_idx = np.random.choice(len(questions['questions']),10)        
    for line in l_idx:
        words = questions['questions'][line]['question'].lower().strip()
        tag = annotations['annotations'][line]['multiple_choice_answer']
        img_id = questions['questions'][line]['image_id']
        h5_id = visual_feat_mapping[str(img_id)]
        img_feat = img_features[h5_id]
        yield Example2(words=[wt2i[x] for x in words.split(" ")],
                      tag=tt2i[tag],
                      img_feat=img_feat,
                      img_id = img_id)

   
test = list(read_test("../data/vqa_questions_test.gzip",
                        "../data/vqa_annotatons_test.gzip",
                        "../data/features/VQA_image_features.h5",
                        "../data/features/VQA_img_features2id.json",
                        "../data/imgid2imginfo.json"))

wt2i = defaultdict(lambda: UNK, wt2i)
ntwords = len(wt2i)
nttags = len(tt2i)
it2w, it2t = {},{}
for key,value in wt2i.items():
    it2w[value] = key
for key,value in tt2i.items():
    it2t[value] = key
    
#%%
with open("../data/imgid2imginfo.json", 'r') as file:
    imgid2info = json.load(file)
    idx = [example.img_id for example in test]
    j=0
    for i in idx:
        url = imgid2info[str(i)]['flickr_url']
        response = requests.get(url)
        if response.status_code == 200:
            with open("./sample{}.jpg".format(j), 'wb') as f:
                f.write(response.content)
        j +=1
        
#%%
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

def predict(model, data):
    """Evaluate a model on a data set."""
    #correct = 0.0


    seqs, tags, image = preprocess(data)
    scores = model(get_variable(seqs), get_image(image))
    _, predictions = torch.max(scores.data, 1)
        #targets = torch.cuda.LongTensor(tags) if CUDA else torch.LongTensor(tags)

        #correct += torch.eq(predictions, targets).sum()

    return seqs, tags, predictions

def vqa(model,data,name):
    plt.figure()
    ques,ans,pred = predict(model,data)
    for i in range(len(ques)):
        if name=='test':
            ques_str = [it2w[q] for q in ques[i]]
            ques_str = ques_str[0:len(data[i].words)]
            ans_str = it2t[ans[i]]
            if (pred[i] in list(it2t.keys())):
                pred_ans = it2t[pred[i]]
            elif (pred[i] in list(itr2t.keys())):
                pred_ans = itr2t[pred[i]]
            else:
                pred_ans = "random prediction"        
            print("Question = {}".format(ques_str))           
            img = mpimg.imread('sample{}.jpg'.format(i))
            plt.imshow(img)
            plt.show()
            print("Correct Answer = {}".format(ans_str))
            print("Predicted Answer = {}".format(pred_ans))
            print("\n")
        else:
            ques_str = [itr2w[q] for q in ques[i]]
            ques_str = ques_str[0:len(data[i].words)]
            ans_str = itr2t[ans[i]]
            if (pred[i] in list(itr2t.keys())):
                pred_ans = itr2t[pred[i]]
            else:
                pred_ans = "random prediction"        
            print("Question = {}".format(ques_str))           
            img = mpimg.imread('train{}.jpg'.format(i))
            plt.imshow(img)
            plt.show()
            print("Correct Answer = {}".format(ans_str))
            print("Predicted Answer = {}".format(pred_ans))
            print("\n")
        
#%%
print("loading saved model")
with open('my_model2.pkl','rb') as f:
	model=pickle.load(f)
#%%
print("Predicting on Training Images")
print("\n")    
vqa(model,train_pred,"train")
print("Predicting on Test Images")
print("\n") 
vqa(model,test,"test")
    


