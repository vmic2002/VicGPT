from datasets import load_dataset, Dataset
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
def main():
    #ds = load_dataset("wikitext", "wikitext-103-v1")
    ds = load_dataset("roneneldan/TinyStories")
    print("Dataset loaded:")
    print(ds)
    print("-"*50)
    """
    print("Printing first few examples:")
    i=0
    for item in ds["train"]:
        print(f"entry {i}")
        print(item["text"])
        print("-"*40)
        i+=1
        if i>2:
            break
    """
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token # gpt2 has no pad_token
    print(f"Tokenizer {tokenizer_name} loaded")
    print(f"eos id:{tokenizer.eos_token_id}, pad: {tokenizer.pad_token_id}")
    ########################################################
    max_seq_len=128 # TODO THIS SHOULD BE INPUT TO THE SCRIPT USING ARGSPARSE
    num_proc = 8 # number of processes for tokenizing dataset
    train_batch_size=32
    val_batch_size=32
    num_epochs=1
    lr = 1e-5
    weight_decay=0.01
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ########################################################

    vicGPT = VicGPT(
        vocab_size=tokenizer.vocab_size,
        num_layers=1,
        embed_dim=512,
        num_heads=4,
        max_seq_len=max_seq_len,
        mlp_hidden_dim=256,
        dropout_prob=0.1
    )
    print("Model to be pre-trained:")
    print(vicGPT)

    def tokenize(examples):
        """
        tokenizer() returns a dict of {"input_ids":..., "attention_mask":...}
        """
        # need to add <eos> token at end of each sequence in case of truncation
        # this way the model will learn to end its generation with <eos>
        result = tokenizer(examples["text"], truncation=True, max_length=max_seq_len-1)#-1 to leave space for <eos>
        result["input_ids"] = [ids + [tokenizer.eos_token_id] for ids in result["input_ids"]]
        result["attention_mask"] = [mask + [1] for mask in result["attention_mask"]]
        return result
    ########################
    ds["train"] = ds["train"].select(range(10)) # TODO FOR TESTING ONLY, comment out
    ds["validation"] = ds["validation"].select(range(10))# TODO FOR TESTING ONLY, COMMENT OUT
    #######################
    print("Tokenizing entire dataset:")
    tokenized_ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=["text"])
    #print(tokenized_ds["train"][0]["input_ids"])
    #print(tokenized_ds["train"][0]["attention_mask"])

    # sanity check, decoding the input ids should be equal to the original text
    decoded_test_string = tokenizer.decode(tokenized_ds["train"][0]["input_ids"][:-1])
    assert decoded_test_string == ds["train"][0]["text"][:len(decoded_test_string)]
    print("Tokenization successfull")



    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    #print(tokenized_ds["train"][0]["input_ids"])

    # need data collator to ensure all sequences in a batch have the same length by padding the shorter ones
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        tokenized_ds["train"],
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        tokenized_ds["validation"],
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator
    )
    optimizer = AdamW(vicGPT.parameters(), lr=lr, weight_decay=weight_decay)
    vicGPT.to(device)
    print(f"Using device {device}")
    
    train_losses = []
    val_losses = []
    print("Starting training") 
    for epoch in range(num_epochs):
        epoch_loss = 0
        vicGPT.train()
        for batch_idx, batch in enumerate(train_loader):
            
            input_ids = batch["input_ids"].to(device) # (batch_size, seq_len)
            attention_mask = batch["attention_mask"] # (batch_size, seq_len) TODO NEED TO PUT ON DEVICE TOO?
            print(f"batch id {batch_idx}")
            #print(input_ids.shape) 
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
            logits = logits.reshape(-1, tokenizer.vocab_size) # (batch_size * seq_len-1, vocab_size)
            loss = F.cross_entropy(logits, labels, ignore_index=-100)
            # TODO, STILL NEED TO VERIFY THAT TRUNCATION + PADDING + ATTENTION MASKS ARE ALL WORKING WELL
            # TODO, FOR THIS COULD TRY WITH A VERY SMALL MAX SEQ LEN AND 1 EXAMPLE 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss/len(train_loader)
        train_losses.append(avg_epoch_loss)
        #TODO ALSO SAVE MODEL WEIGTHS EVERY COUPLE EPOCHS, maybe only keep last 3 saves
        # TODO vicGPT.eval(), do validation loss
        # val_loss = 0
        #with torch.no_grad():
    print(train_losses)
    print("Done!")
    # TODO SAVE MODEL, MAYBE THE ONE WITH LOWEST VAL LOSS?
            
if __name__ == "__main__":
    main()
