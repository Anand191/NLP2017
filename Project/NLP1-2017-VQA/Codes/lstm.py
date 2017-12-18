#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:49:29 2017

@author: anand
"""
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Deep CBOW model
    """

    def __init__(self, vocab_size, embedding_dim, img_f, output_dim, lstm_dim,num_lstms,hidden_dims, activation):
        """
        :param vocab_size: Vocabulary size of the training set.
        :param embedding_dim: The word embedding dimension.
        :param output_dim: The output dimension, ie the number of classes.
        :param hidden_dims: A list of hidden layer sizes. Default: []
        :param transformations: A list of transformation functions.
        """
        super(LSTM, self).__init__()
        self.rnn_hidden = lstm_dim
        self.num_layers = len(hidden_dims)
        self.activation = activation
        self.units = num_lstms
        
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim,self.rnn_hidden,self.units,batch_first=True) 

        self.linears = {}
        if (self.num_layers == 0):
            self.linear1 = nn.Linear(img_f + self.rnn_hidden, output_dim)
            self.linears[0] = self.linear1
        else:
            self.linear1 = nn.Linear(img_f+ self.rnn_hidden, hidden_dims[0])
            self.linears[0] = self.linear1
            for i in range(1, self.num_layers):
                l = "self.linear" + str(i+1)
                exec(l + " = nn.Linear(hidden_dims[i-1], hidden_dims[i])")
                exec("self.linears[i] = " + l)
            l = "self.linear" + str(self.num_layers + 1)
            exec(l + " = nn.Linear(hidden_dims[self.num_layers-1], output_dim)")
            exec("self.linears[self.num_layers] = " + l)

    def forward(self, words, image):
        embeds = self.embeddings(words)
        sequence,_ = self.rnn(embeds)
        #print(sequence.size())
        h = torch.sum(sequence, 1)
        #print(h.size())
        h = torch.cat([image, h], dim=1)
        #print(h.size())
        if(self.num_layers == 0):
            h = self.linears[0](h)
        else:
            for i in range(self.num_layers):
                h = self.activation(self.linears[i](h))
            h = self.linears[self.num_layers](h)
        return h