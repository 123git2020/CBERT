#!/usr/bin/env python
from __future__ import print_function

import argparse
import json

from random import shuffle
import numpy as np
import torch
from torch.nn import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def get_x_y(train_txt, num_classes, word2vec_len, input_size, word2vec,indices):

    #read in lines
	train_lines = open(train_txt, 'r').readlines()
	train_lines.pop(0)      				# 移除标题行

	train_lines=[train_lines[i] for i in indices]
	shuffle(train_lines)
	num_lines = len(train_lines)

	#initialize x and y matrix
	x_matrix = None
	y_matrix = None

	try:
		x_matrix = np.zeros((num_lines, input_size, word2vec_len))
	except:
		print("Error!", num_lines, input_size, word2vec_len)
	y_matrix = np.zeros((num_lines, num_classes))

	#insert values
	for i, line in enumerate(train_lines):   #遍历每个句子

		sentence = line[:-3]
		label = int(line[-2])

		#insert x
		words = sentence.split(' ')         
		words = words[:input_size]          #cut off if too long
		for j, word in enumerate(words):    #遍历每个词，获得其词向量
			if word in word2vec:
				x_matrix[i, j, :] = word2vec[word]

		#insert y
		y_matrix[i][label] = 1.0            #将其标签对应的下标置1,one-hot

	return x_matrix, y_matrix

def to_dataset(x,y,batch_size):
	x=torch.tensor(x).float().cuda()
	y=torch.tensor(y).float().cuda()
	data=TensorDataset(x,y)
	sampler = RandomSampler(data)
	dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

	return data,dataloader

def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)

class RNNModel(torch.nn.Module):

	def __init__(self, num_classes, drop_rate=0.5) -> None:
		super(RNNModel,self).__init__()
		
		self.layer_0=GRU(300,64,dropout=drop_rate,bidirectional=True,batch_first=True)    #两层双向GRU
		self.layer_1=GRU(128,32,dropout=drop_rate,bidirectional=True,batch_first=True)
		self.classifier=Sequential(                      
         ReLU(),
         Linear(64,32),
         ReLU(), 
         Linear(32,num_classes)    
    	)


	def forward(self,x):

		seq0,_=self.layer_0(x)					#前向过程
		_,h1=self.layer_1(seq0)					#最后一个隐藏态作为线性层输入
		h1=torch.cat((h1[0],h1[1]),1)			#bidirection的隐藏态要合并
		
		logits=self.classifier(h1)
		preds=Softmax(dim=1)(logits)

		return preds
		