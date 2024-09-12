import torch
import torch.nn as nn
import torch.nn.functional as F
# import utils

"""
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
    def forward(self, x:torch.Tensor):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.input_dim ** 0.5)
        attention = F.softmax(scores)
        weighted = torch.matmul(attention, values)
        return weighted

"""

class SelfAttention(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttention, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x:torch.Tensor, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)
        #return output, attention_weights
        return output
