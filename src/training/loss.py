import torch.nn.functional as F

def training_loss(vicGPT, input_ids, attention_mask, vocab_size):
    logits = vicGPT(input_ids) # (batch_size, seq_len, vocab_size)
    # now need to generate labels
    labels = input_ids.clone()
    #print(attention_mask)
    labels[attention_mask == 0] = -100 # set labels to -100 at padding locations to not include in loss
    # need to shift labels and logits so that logits[i] predict labels[i+1] (next token prediction)
    labels = labels[:, 1:] # labels at positions 1 to seq_len-1
    logits = logits[:, :-1, :] # logits at positions 0 to seq_len-2
    # need to reshape for cross_entropy
    labels = labels.reshape(-1) # (batch_size * seq_len-1)
    logits = logits.reshape(-1, vocab_size) # (batch_size * seq_len-1, vocab_size)
    #print("labels:")
    #print(labels)
    #print("logits:")
    #print(logits)
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
    return loss
