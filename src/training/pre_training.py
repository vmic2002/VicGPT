from datasets import load_dataset, Dataset
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb

def pre_training_loss(vicGPT, input_ids, attention_mask, vocab_size):
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

def main():
    #ds = load_dataset("wikitext", "wikitext-103-v1")
    dataset_name = "roneneldan/TinyStories"
    ds = load_dataset(dataset_name)
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
    max_seq_len=1024 # TODO THIS SHOULD BE INPUT TO THE SCRIPT USING ARGSPARSE
    num_proc = 8 # number of processes for tokenizing dataset
    train_batch_size=32
    val_batch_size=32
    num_epochs=5
    lr = 1e-5
    weight_decay=0.01
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    num_layers=1
    embed_dim=512
    num_heads=4
    mlp_hidden_dim=256
    dropout_prob=0.1
    ########################################################

    vicGPT = VicGPT(
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout_prob=dropout_prob
    )
    print("Model to be pre-trained:")
    #print(vicGPT)

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
    ds["train"] = ds["train"].select(range(50)) # TODO FOR TESTING ONLY, comment out
    ds["validation"] = ds["validation"].select(range(20))# TODO FOR TESTING ONLY, COMMENT OUT
    #######################
    print("Tokenizing entire dataset:")
    tokenized_ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=["text"])
    
    ##########################################
    # FOR TESTING
    #for idx, example in enumerate(tokenized_ds["train"]):
    #    print(f"Tokenized Example {idx}, with {len(example['input_ids'])} tokens")
    #    print(example["input_ids"])
    #    print(example["attention_mask"])
    ##########################################
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
    run = wandb.init(project="vicGPT",
        config={
            "learning_rate": lr,
            "dataset": dataset_name,
            "tokenizer": tokenizer_name,
            "epochs": num_epochs,
            "num_layers": num_layers,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "mlp_hidden_dim": mlp_hidden_dim,
            "dropout_prob": dropout_prob 
        }        
    )
    print("Starting training") 
    for epoch in range(num_epochs):
        epoch_loss = 0
        vicGPT.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch}")):
            
            input_ids = batch["input_ids"].to(device) # (batch_size, seq_len)
            attention_mask = batch["attention_mask"].to(device) # (batch_size, seq_len)
            #print("Batch input_ids:")
            #print(input_ids)
            #print("Batch attention mask:")
            #print(attention_mask)
            #print(f"batch id {batch_idx}")
            #print(f"input ids shape: {input_ids.shape}") 
            #####################
            loss = pre_training_loss(vicGPT, input_ids, attention_mask, tokenizer.vocab_size)
            #print(f"loss: {loss}")
            ####################### 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss/len(train_loader)
        train_losses.append(avg_epoch_loss)
        #TODO ALSO SAVE MODEL WEIGTHS EVERY COUPLE EPOCHS, maybe only keep last 3 saves 
        vicGPT.eval() # validation
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                input_ids = batch["input_ids"].to(device) # (batch_size, seq_len)
                attention_mask = batch["attention_mask"].to(device) # (batch_size, seq_len)
                #print("Batch input_ids:")
                #print(input_ids)
                #print("Batch attention mask:")
                #print(attention_mask)
                #print(f"batch id {batch_idx}")
                #print(f"input ids shape: {input_ids.shape}")
                #####################
                loss = pre_training_loss(vicGPT, input_ids, attention_mask, tokenizer.vocab_size)
                #######################
                val_loss += loss.item()
            avg_epoch_loss = val_loss/len(val_loader)
            val_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Train loss: {train_losses[-1]} Validation Loss: {val_losses[-1]}") 
        run.log({"train_loss": train_losses[-1], "val_loss":val_losses[-1]})
    print("-"*50)
    print("Done training")
    run.finish()
    
    print(train_losses)
    print(val_losses)
    print("Done!")
    # TODO SAVE MODEL, MAYBE THE ONE WITH LOWEST VAL LOSS?
            
if __name__ == "__main__":
    main()
