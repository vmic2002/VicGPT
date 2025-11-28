import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer_block import TransformerBlock

class VicGPT(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads, max_seq_len, mlp_hidden_dim, dropout_prob=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_prob = dropout_prob
        
        self.transformer_blocks = nn.ModuleList([ 
            TransformerBlock(embed_dim, num_heads, max_seq_len, mlp_hidden_dim, dropout_prob)
            for _ in range(num_layers)
        ])

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(max_seq_len, embed_dim) # learned positional encodings
        self.language_head = nn.Linear(embed_dim, vocab_size)  

    def forward(self, x):
        # x is list of token ids, not yet embedded
        batch_size, seq_len = x.shape
        x = self.token_embedding(x) # (batch_size, seq_len, embed_dim)
        x = x + self.positional_encoding(torch.arange(seq_len, device=x.device))
        for i, t_block in enumerate(self.transformer_blocks):
            x = t_block(x)
        x = self.language_head(x) # (batch_size, seq_len, vocab_size) --> logits!
        return x

    @staticmethod 
    def top_p_sampling(logits, top_p):
        # logits.shape --> (batch_size, vocab_size)
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token (shift right to keep the first token above threshold)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False # always keep most likely token
        # Set removed tokens to -inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        # Scatter back to original order
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return logits_filtered 

    @staticmethod
    def top_k_sampling(logits, top_k):
        # logits.shape --> (batch_size, vocab_size)
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        # Set all non-top-k logits to -inf
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)
        return logits_filtered

    def generate(self, x, max_new_tokens=200, temperature=1.0, top_p=0.9, top_k=None, eos_token_id=None):
        # x is list of input token ids
        # this function autoregressively samples new tokens, and returns the prompt + new output
        # the tokenizer is used before the function to encode the prompt and after to decode the output
        batch_size, seq_len = x.shape
        assert batch_size == 1, "batch_size for generation should be 1"
        num_new_tokens = 0
        # keep generating until max new tokens is reached or end of sequence (EOS)
        while num_new_tokens < max_new_tokens: 
            logits = self.forward(x)
            last_token_logits = logits[:,-1,:]/temperature # (batch_size=1, vocab_size) last token in sequence used to generate new token
            if top_k:
                last_token_logits = self.top_k_sampling(last_token_logits, top_k)
            elif top_p:
                last_token_logits = self.top_p_sampling(last_token_logits, top_p)         
            next_token_prob_distribution = F.softmax(last_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_prob_distribution, num_samples=1) # (batch_size=1, 1)
            # concat next token into the current context
            x = torch.cat([x, next_token_id], dim=-1)
            num_new_tokens += 1 
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
        return x

# TODO, NEXT STEP MIGHT BE TO WRITE A TOKENIZER CLASS, MAKE SURE HUGGING FACE TOKENIZER AND OUR OWN HAVE SAME FUNCTIONS SO THEY CAN BOTH BE USED EASILY
# TODO, COULD IMPLEMENT A GENERATE func THAT USES A KV CACHE!
# TODO NEED TO UNIT TEST THE TOP K AND TOP P SAMPLING FUNCTIONS
