import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadMaskedSelfAttention

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, mlp_hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_prob = dropout_prob
        self.attention = MultiHeadMaskedSelfAttention(embed_dim, num_heads, max_seq_len)
        self.mlp1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.mlp2 = nn.Linear(mlp_hidden_dim, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # batch_size, seq_len, embed_dim = x.shape
        norm_x = self.layer_norm1(x) # pre norm, normalization done before attention is best standard
        attention_output = self.attention(norm_x)
        attention_output = self.dropout(attention_output)
        residual = x + attention_output
        mlp1_input = self.layer_norm2(residual) # second layer norm with residual connection
        mlp2_output = self.mlp2(F.gelu(self.mlp1(mlp1_input)))
        mlp2_output = self.dropout(mlp2_output)
        output = mlp2_output + residual # residual connection
        return output

