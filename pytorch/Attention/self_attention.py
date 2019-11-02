import torch.nn as nn
import torch
import math
from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
#Refer scaled_dot_product_attention.png
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        assert q.size(-1) == d_k

        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature)) # (Batch, Seq, Seq) #bmm : batch matrix multiplication
        # we get an attention score between each position in the sequence
        # for each batch
        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / math.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attn = torch.exp(attn)
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(-1, keepdim=True) #softmax
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # (Batch, Seq, Feature)
        return output



class AttentionHead(nn.Module):
    
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        #we will assume queries,key abd values all have same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)
    
    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x
        
        


