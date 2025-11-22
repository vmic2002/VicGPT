import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sqrt_dk = self.head_dim ** 0.5
        self.kqv = nn.Linear(embed_dim, embed_dim*3)
        self.max_seq_len = max_seq_len
       
        self.register_buffer( # use register_buffer for non-learnable tensors that are part of model
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        # for max_seq_len of 3, causal mask looks like
        # _____________________
        # | False True  True  |
        # | False False True  |
        # | False False False |
        # _____________________

    def forward(self, x):
        """
        Attention = softmax(Q @ K.T / d_k) @ V
        """
        batch_size, seq_len, embed_dim = x.shape
        #  no need to assert embed_dim == self.embed_dim since kqv layer will fail if the dimensions do not match
        kqv = self.kqv(x) # (batch_size, seq_len, embed_dim*3) --> each embeddings' k,q,v are concatenated into 1 vector
        # need to reshape to structure the k,q,v for each attention head
        kqv = kqv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim) # 3 for each k,q,v
        # need to permute to access the k,q,v for each attention head
        kqv = kqv.permute(2, 0, 3, 1, 4) # --> new shape is (3, batch_size, num_heads, seq_len, head_dim)
        K, Q, V = kqv[0], kqv[1], kqv[2] # these contain the K, Q, V of all attention heads 
        # each K, Q, V has shape (batch_size, num_heads, seq_len, head_dim)
        K_transposed = K.transpose(-2,-1) # (batch_size, num_heads, head_dim, seq_len)
        attention_scores = Q @ K_transposed / self.sqrt_dk # (batch_size, num_heads, seq_len, seq_len)
        # apply mask so every token only attends to previous ones
        # this will set every value in the top right corner to -inf, excluding the diagonal
        masked_attention_scores = attention_scores.masked_fill(self.causal_mask[:seq_len, :seq_len], float('-inf')) #
        attention_weights = F.softmax(masked_attention_scores, dim=-1) # -inf values are set to zero
        output_per_head = attention_weights @ V # (batch_size, num_heads, seq_len, head_dim)
        output_per_head = output_per_head.transpose(1, 2) # (batch_size, seq_len, num_heads, head_dim)
        output = output_per_head.reshape(batch_size, seq_len, embed_dim)
        return output
        
