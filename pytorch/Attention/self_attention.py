import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.01):
        super().__init__()
        
            






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
        
        


