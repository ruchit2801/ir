import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


LETTER_GRAM_SIZE = 3
WINDOW_SIZE = 3
TOTAL_LETTER_GRAMS = 30000
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS

K = 300
L = 128
J = 4
FILTER_LENGTH = 1

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM, self).__init__()
        
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        
    def forward(self, q, pos, negs):
        """
        In this step, we transform each word vector with WORD_DEPTH dimensions into its
        convolved representation with K dimensions. K is the number of kernels/filters
        being used in the operation. Essentially, the operation is taking the dot product
        of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        That is, h_Q = tanh(W_c • l_Q + b_c). Note: the paper does not include bias units.
        """
        q = q.transpose(1,2)
        
        q_c = F.tanh(self.query_conv(q))
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1, 2)
        
        q_s = F.tanh(self.query_sem(q_k))
        q_s = q_s.resize(L)
        
        pos = pos.transpose(1,2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1,2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = pos_s.resize(L)


        negs = [neg.transpose(1,2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1,2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [neg_s.resize(L) for neg_s in neg_ss]


        dots = [q_s.dot(pos_s)]
        dots = dots + [q_s.dot(neg_s) for neg_s in neg_ss]
        dots = torch.stack(dots)


        with_gamma = self.learn_gamma(dots.resize(J+1, 1, 1))
        return with_gamma
        
model = CDSSM()


# Preorocessing goes from here 


