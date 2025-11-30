from datasets import load_dataset, Dataset
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer
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
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_len)
    ########################
    ds["train"] = ds["train"].select(range(10)) # TODO FOR TESTING ONLY, comment out
    ds["validation"] = ds["validation"].select(range(10))# TODO FOR TESTING ONLY, COMMENT OUT
    #######################
    print("Tokenizing entire dataset:")
    tokenized_ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=["text"])
    print(tokenized_ds)


    # sanity check, decoding the input ids should be equal to the original text
    decoded_test_string = tokenizer.decode(tokenized_ds["train"][0]["input_ids"])
    assert decoded_test_string == ds["train"][0]["text"][:len(decoded_test_string)]
    print("Tokenization successfull")



    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    #print(tokenized_ds["train"][0]["input_ids"])

    train_loader = DataLoader(
        tokenized_ds["train"],
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        tokenized_ds["validation"],
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0
    )

    #print(train_loader[0])
    optimizer = AdamW(vicGPT.parameters(), lr=lr, weight_decay=weight_decay)
    vicGPT.to(device)
    print(f"Using device {device}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        vicGPT.train()
        for batch_idx, batch in enumerate(train_loader):
            
            input_ids = batch["input_ids"].to(device) # (batch_size, seq_len)
            print(f"batch id {batch_idx}")
            #print(input_ids.shape) 
            logits = vicGPT(input_ids) # (batch_size, seq_len, vocab_size)
            print(logits)
            print(logits.shape)
            # TODO loss = F.cross_entropy()
            return
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
        
    print("Done!")
    # TODO SAVE MODEL, MAYBE THE ONE WITH LOWEST VAL LOSS?
            
if __name__ == "__main__":
    main()
# USE DATALOADER AND BATCHING
# TRAIN FOR EVERY EPOCH
# also every var, including batch size should be export in the pre_training.sh in the root directory
print("done")
