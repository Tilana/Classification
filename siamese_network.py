import fastai
from fastai.text import *
import html

import json
import html
import re
import pickle
from collections import Counter
import random
import pandas as pd
import numpy as np
from pathlib import Path
import sklearn
from sklearn import model_selection
from functools import partial
from collections import Counter, defaultdict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils 
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import dataset, dataloader
import torch.optim as optim
import torch.nn.functional as F

import time
import math
import sys
import data

import pdb

snli_root = './data/snli_1.0/'
token_files = './data/tokens/'


class SiameseDataLoader():
    def __init__(self, sentence_pairs, pad_val, batch_size=32):
        self.sentence_pairs = sentence_pairs
        self.batch_size = batch_size
        self.index = 0
        self.pad_val = pad_val
     
    def shuffle(self):
        def srtfn(x):
            return x[:, -1] + random.randint(-5, 5)
        
        order = np.argsort(srtfn(self.sentence_pairs))
        self.sentence_pairs = self.sentence_pairs[order]
        
    def __iter__(self):
        return self
    
    def fill_tensor(self, sentences, max_len):
        data = np.zeros((max_len, len(sentences)), dtype=np.long)
        data.fill(self.pad_val)
        
        for i, s in enumerate(sentences): 
            start_idx = max_len - len(s)
            for j, p in enumerate(s):
                data[:,i][start_idx+j] = p
            
        return torch.LongTensor([data.tolist()]).cuda()
     
    def batch(self):
        return self.index//self.batch_size
    
    def __len__(self):
        return len(self.sentence_pairs)//self.batch_size
    
    def __next__(self):
        #how many examples to ananlyise for this round
        num = min(self.batch_size, len(self.sentence_pairs) - self.index)
        
        if num < 1:
            raise StopIteration  # signals "the end"
            
        #collect the sentences
        max_len_a = 0
        max_len_b = 0
        first = []
        second = []
        labels = torch.LongTensor(num)
        
        for i in range(num):
            a, b, l, _ = self.sentence_pairs[self.index + i]
            
            if len(a) > max_len_a:
                max_len_a = len(a)
            
            if len(b) > max_len_b:
                max_len_b = len(b)
            
            first.append(a)
            second.append(b)
            labels[i] = l 
            
        self.index += num
        
        first = self.fill_tensor(first, max_len_a)
        second = self.fill_tensor(second, max_len_b)
        return (first.cuda(),
                (first != self.pad_val).cuda(),
                second.cuda(),
                (second != self.pad_val).cuda(),
                labels.cuda()
               )


itos = pickle.load(open(f'{token_files}itos.pkl', 'rb'))
stoi = defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
vocab_size = len(itos)
pad_tok = stoi['_pad_']

sentence_pairs_train = np.load(f'{token_files}snli_tok_train.npy')
sentence_pairs_dev = np.load(f'{token_files}snli_tok_dev.npy')
sentence_pairs_test = np.load(f'{token_files}snli_tok_test.npy')

def print_sentence(s):
    sentence = ""
    for tok in s:
        sentence += " "+itos[tok]
    print(sentence)

print_sentence(sentence_pairs_train[0][0])
print_sentence(sentence_pairs_train[0][1])

print_sentence(sentence_pairs_dev[0][0])
print_sentence(sentence_pairs_dev[0][1])

print_sentence(sentence_pairs_test[0][0])
print_sentence(sentence_pairs_test[0][1])

training_data = SiameseDataLoader(sentence_pairs_train, pad_tok)
for batch in training_data:
    sentences = batch[0][0]
    masks = batch[1][0]
    for sentence, mask in zip(sentences.transpose(1,0), masks.transpose(1,0)):
        for tok in torch.masked_select(sentence, mask):
            print(itos[int(tok)], end=' ')
        print("")
    break

# sentences are in the form [sentence_length, batch_size, embedding_size]
# masks are in the form [sentence_length, batch_size])
sentence_length = 5
batch_size = 3
embedding_size = 4

out = torch.zeros((batch_size, embedding_size))
sentences = torch.tensor([ 
                    [[1,1,1,1], [4,4,4,4], [7,7,7,7]],
                    [[2,2,2,2], [5,5,5,5], [8,8,8,8]],
                    [[0,0,0,0], [6,6,6,6], [9,9,9,9]],
                    [[0,0,0,0], [0,0,0,0], [10,10,10,10]],
                    [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
                    ]).float()

#sentences.shape == [5, 3, 4]

masks = torch.tensor([[[1,1,1], [1,1,1], [0,1,1], [0,0,1], [0,0,0]]]).byte()
#masks.shape == [1, 5, 3]

for i, sentence, mask in zip(range(batch_size), sentences.permute((1,0,2)), masks.squeeze().permute(1,0)):
    mask = mask.unsqueeze(1)
    selected = torch.masked_select(sentence, mask)
    selected = torch.reshape(selected, (-1, embedding_size))
    print(selected)
    max = torch.max(selected, 0)[0]
    print(max)
    out[i] = torch.mean(selected, 0)
print(out)


class SiameseClassifier(nn.Module):
    
    def __init__(self, encoder, linear):
        super().__init__()
        self.encoder = encoder
        self.linear = linear
    
    def pool(self, x, masks, is_max):
        #x.shape = sentence length, batch size, embedding size
        #mask.shape = [1, sentence length, batch size]
        
        embedding_size = x.shape[2]
        batch_size = x.shape[1]
        out = torch.zeros((batch_size, embedding_size)).cuda()
        masks = masks.squeeze()
        #print(f'shapes: x {x.shape}, masks {masks.shape}, out {out.shape}')
        
        #shapes: x torch.Size([7, 32, 400]), mask torch.Size([7, 32]), out torch.Size([32, 400])
                
        for i, hidden, mask in zip(range(batch_size), x.permute((1,0,2)), masks.permute(1,0)):
            mask = mask.unsqueeze(1)
            selected = torch.masked_select(hidden, mask)
            selected = torch.reshape(selected, (-1, embedding_size))
            if is_max:
                max_pool = torch.max(selected, 0)[0]
                out[i] = max_pool
            else:
                mean_pool = torch.mean(selected, 0)
                out[i] = mean_pool

        return out

    def pool_outputs(self, output, mask):
        avgpool = self.pool(output, mask, False)
        maxpool = self.pool(output, mask, True)
        last = output[-1]
        return torch.cat([last, maxpool, avgpool], 1)
        
    def forward_once(self, input, mask):
        raw_outputs, outputs = self.encoder(input)
        out = self.pool_outputs(outputs[-1], mask)
        return out
    
    def forward(self, in1, in1_mask, in2, in2_mask):
        u = self.forward_once(in1, in1_mask)
        v = self.forward_once(in2, in2_mask)
        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        out = self.linear(features)
        return out 
        
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()

class LinearClassifier(nn.Module):
    def __init__(self, layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([LinearBlock(layers[i], layers[i + 1], dropout) for i in range(len(layers) - 1)])
        
    def forward(self, input):
        x = input
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x

#these are the values used for the original LM
em_sz, nh, nl = 400, 1150, 3
bptt = 70
max_seq = bptt * 20
cats = 3


log_interval = 1000
criterion = nn.CrossEntropyLoss()
#criterion = nn.CosineEmbeddingLoss()

def evaluate(model, data_loader):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    
    total_loss = 0.
    num_correct = 0
    total = 0 
    
    for a, a_mask, b, b_mask, l in data_loader:
        
        model.reset()
        out = model(a.squeeze(), a_mask, b.squeeze(), b_mask)
        loss = criterion(out, l.squeeze())
        total += l.size(0)
        total_loss += l.size(0) * loss.item()
        num_correct += np.sum(l.data.cpu().numpy() == np.argmax(out.data.cpu().numpy(), 1))
        
    return (total_loss / total, num_correct / total)

def train(model, data_loader, optimizer):
    # Turn on training mode which enables dropout.
    start_time = time.time()
    model.train() 
    
    total_loss = 0.
    num_correct = 0
    total = 0 
        
    for a, a_mask, b, b_mask, l in data_loader:
        optimizer.zero_grad()
        
        model.reset()
        #torch.Size([1, 7, 32])
        
        out = model(a.squeeze(), a_mask, b.squeeze(), b_mask)
        loss = criterion(out, l.squeeze())
        total += l.size(0)
        total_loss += l.size(0) * loss.item()
        num_correct += np.sum(l.data.cpu().numpy() == np.argmax(out.data.cpu().numpy(), 1))
        
        loss.backward()
        optimizer.step()

        batch = data_loader.batch()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / total
            elapsed = time.time() - start_time
            batches = len(data_loader)
            ms = elapsed * 1000 / log_interval
            print(f'| {batch:5d}/{batches:5d} batches', end=" ")
            print(f'| ms/batch {ms:5.2f} | loss {cur_loss:5.4f} acc {num_correct / total}')
            #print(f'| ms/batch {ms:5.2f} | loss {cur_loss:5.4f}')
            total_loss = 0
            total = 0
            num_correct = 0
            start_time = time.time()


best_loss = 100
def training_loop(model, epochs, optimizer, scheduler = None):
    
    global best_loss
    for epoch in range(epochs):

        print(f'Start epoch {epoch:3d} training with lr ', end="")
        for g in optimizer.param_groups:
            print(g['lr'], end=" ")
        print("")
        
        training_data = SiameseDataLoader(sentence_pairs_train, pad_tok)
        training_data.shuffle()

        epoch_start_time = time.time()
        
        train(model, training_data, optimizer)
        if scheduler != None:
            scheduler.step()

        dev_data = SiameseDataLoader(sentence_pairs_dev, pad_tok)
        val_loss, accuracy = evaluate(model, dev_data)

        delta_t = (time.time() - epoch_start_time)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {delta_t:5.2f}s | valid loss {val_loss:5.2f} accuracy {accuracy} learning rates')
        for g in optimizer.param_groups:
            print(g['lr'])
        print('-' * 89)

        if val_loss < best_loss:
            best_loss = val_loss
            with open(f'./siamese_model{val_loss:0.2f}{accuracy:0.2f}.pt', 'wb') as f:
                torch.save(siamese_model, f)


from scipy.signal import butter, filtfilt
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def plot_loss(losses):
    plt.semilogx(losses[:,0], losses[:,1])
    plt.semilogx(losses[:,0], butter_lowpass_filtfilt(losses[:,1], 300, 5000))
    plt.show()

def find_lr(model, model_to_optim, data_loader):
    losses = []
    model.train() 
    criterion = nn.CrossEntropyLoss()
    lr = 0.00001
    for a, b, l in data_loader:
        optimizer = optim.SGD(model_to_optim.parameters(), lr=lr)
        #optimizer = optim.Adam(model_to_optim.parameters(), lr=lr)
        optimizer.zero_grad()
        
        model.reset()
        a, b, l = Variable(a), Variable(b), Variable(l)
        out = model(a.squeeze(), b.squeeze())
        loss = criterion(out, l.squeeze())
        
        los_val = loss.item()
        losses.append((lr, los_val))
        if los_val > 5:
            break
        
        loss.backward()
        optimizer.step()
        
        lr *= 1.05
    losses = np.array(losses)
    #plot_loss(losses)
    return losses


SNLI_LM = torch.load("snli_language_model.pt")

dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.4
SNLI_encoder = MultiBatchRNN(bptt, max_seq, vocab_size, em_sz, nh, nl, pad_tok, dropouti=dps[0], wdrop=dps[2], dropoute=dps[3], dropouth=dps[4])

SNLI_encoder.load_state_dict(SNLI_LM[0].state_dict())

#2 pooled vectors, of 3 times the embedding size
siamese_model = SiameseClassifier(SNLI_encoder, LinearClassifier(layers=[em_sz*3*4, nh, 3], dropout=0.4)).cuda()

dev_data = SiameseDataLoader(sentence_pairs_dev, pad_tok)
losses = find_lr(siamese_model, siamese_model, dev_data)
plot_loss(np.array(losses))



