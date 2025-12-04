from datasets import load_dataset, Dataset
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import argparse
from .loss import training_loss
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import re
import json

def train(args):
    pre_training = args.pre_training
    sft = args.sft
    dataset_name = args.dataset_name
    tokenizer_name = args.tokenizer_name
    name_output_model = args.name_output_model
    max_seq_len = args.max_seq_len
    num_proc = args.num_proc
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    num_epochs = args.num_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    mlp_hidden_dim = args.mlp_hidden_dim
    dropout_prob = args.dropout_prob

    #################
    # TODO REMOVE
    if sft:
        print("SFT not implemented yet")
        return
    #################
         
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token # gpt2 has no pad_token
    print(f"Tokenizer {tokenizer_name} loaded")
    print(f"bos id: {tokenizer.bos_token_id}, eos id:{tokenizer.eos_token_id}, pad: {tokenizer.pad_token_id}")
    ########################################################
        ########################################################

    
    #print("Model to be pre-trained:")
    #print(vicGPT)

    def tokenize(examples):
        """
        tokenizer() returns a dict of {"input_ids":..., "attention_mask":...}
        """
        # TODO  for SFT: NEED TO MASK PROMPT AND NOT ANSWER
        # add <bos> and <eos> token at beginning and end of each sequence
        # this way the model will learn to start it's generation with <bos> end its generation with <eos>
        if sft:
            # for SFT, we truncate because each sample should fit in the same context window
            result = tokenizer(examples["text"], truncation=True, max_length=max_seq_len-2)#-2 to leave space for <bos> and <eos>
        if pre_training:
            # for pre_training we do not truncate because we pack the data anyway
            result = tokenizer(examples["text"], truncation=False)
        result["input_ids"] = [[tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id] for ids in result["input_ids"]]
        result["attention_mask"] = [[1] + mask + [1] for mask in result["attention_mask"]]
        return result

    ########################
     
    ds["train"] = ds["train"].select(range(1)) # TODO FOR TESTING ONLY, comment out
    ds["validation"] = ds["validation"].select(range(1))# TODO FOR TESTING ONLY, COMMENT OUT
    MAX_CHARS = 80
    ds["train"] = ds["train"].map(lambda x: {"text": x["text"][:MAX_CHARS]})
    ds["validation"] = ds["validation"].map(lambda x: {"text": x["text"][:MAX_CHARS]})
    print("ds train before tokenization:")
    print(ds["train"]["text"])
    
    #######################
    print("Tokenizing entire dataset:")
    tokenized_ds = ds.map(tokenize, batched=True, num_proc=num_proc, remove_columns=["text"])
    #print("TOKENIZED DATASET:")
    #print(tokenized_ds["train"]["input_ids"])
    #print("-"*40)
    ##########################################
    # FOR TESTING
    #for idx, example in enumerate(tokenized_ds["train"]):
    #    print(f"Tokenized Example {idx}, with {len(example['input_ids'])} tokens")
    #    print(example["input_ids"])
    #    print(example["attention_mask"])
    ##########################################
    # sanity check, decoding the input ids should be equal to the original text
    decoded_test_string = tokenizer.decode(tokenized_ds["train"][0]["input_ids"][1:-1])
    assert decoded_test_string == ds["train"][0]["text"][:len(decoded_test_string)]
    print("Tokenization successfull")


    train_total_tokens = 0
        #print(tokenized_ds["train"][0]["input_ids"])
    if sft:
        print("For SFT, we use data collator to padd shorter sequences in the same batch")
        # need data collator to ensure all sequences in a batch have the same length by padding the shorter ones
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        # TODO, ALSO TRACK NUMBER OF TRAINING TOKENS (WITH ATTENTION MASK 1)
        # TODO train_total_tokens = ???
    if pre_training:
        print("For pre_training, we pack the tokens to maximize the usage of the context window") 
        def pack(ds):
            train_input_ids = np.concatenate(ds["input_ids"])
            assert len(train_input_ids) >= max_seq_len, f"Training data too small: {len(train_input_ids)} tokens < {max_seq_len} max_seq_len"
            total_length = len(train_input_ids) - len(train_input_ids) % max_seq_len
            train_input_ids = train_input_ids[:total_length] #Discard extra tokens from end of the list so number of tokens is exactly divisible by max_seq_length so there is no need for padding 
            train_input_ids_reshaped = train_input_ids.reshape(-1, max_seq_len).astype(np.int32)
            # For pretraining, attention_mask is all 1s (no padding within sequences)
            train_attention_masks = np.ones_like(train_input_ids_reshaped)
             
            #print(f"Actual shape after reshape: {train_input_ids_reshaped.shape}")
            #print(f"PACKED (showing all rows):")
            #for i, row in enumerate(train_input_ids_reshaped):
            #    print(f"Row {i}: {row}")    

            result = Dataset.from_dict({
                "input_ids": train_input_ids_reshaped.tolist(),
                "attention_mask": train_attention_masks.tolist()
            })
            return result

        tokenized_ds["train"] = pack(tokenized_ds["train"])
        tokenized_ds["validation"] = pack(tokenized_ds["validation"])

        print(f"PACKED train data shape: ({len(tokenized_ds['train'])},{max_seq_len})")
        train_total_tokens = len(tokenized_ds["train"]) * max_seq_len
        print(f"train total tokens: {train_total_tokens}")


        data_collator = None 
   
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

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



    vicGPT = VicGPT(
        vocab_size=tokenizer.vocab_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout_prob=dropout_prob
    )
    print("Model to train:")
    print(vicGPT)
    total_params = sum(p.numel() for p in vicGPT.parameters())
    trainable_params = sum(p.numel() for p in vicGPT.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    vicGPT.to(device)
    optimizer = AdamW(vicGPT.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    # TODO IF SFT, NEED TO CHECK IF MODEL AT SFT/NAMEOUTPUTMODEL EXISTS, IF IT DOES NEED TO LOAD FROM THERE
    load_dir = f"models/pre_trained/{name_output_model}"
    os.makedirs(load_dir, exist_ok=True)
    if pre_training:
        save_dir = load_dir # load and save models from the same pre_trained dir
        # save args as json to the save dir since it contains the hyperparams
        args_dict = vars(args)
        #print("args dict:::::::::::")
        #print(args_dict)
        config_path = os.path.join(save_dir, "config.json")
        # Only write if the file does not already exist
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(args_dict, f, indent=4)
            print(f"Saved config to {config_path}")
    if sft:
        save_dir = f"models/sft/{name_output_model}"
        # TODO FOR SFT NEED TO CREATE DIR ETC, and save config.json
        # TODO for sft however, if there is an entry in sft dir, we load it. but if there is not, we load from pre_trained dir (since we take the pre_trained model and do sft). for sft we always save in sft dir of course

    
    

    start_epoch = 0
    best_val_loss = float("inf")

    # === Check for existing checkpoints ===
    checkpoint_files = sorted(
        glob(os.path.join(load_dir, "checkpoint_epoch*.pt")),
        key=lambda x: int(re.search(r"checkpoint_epoch(\d+).pt", x).group(1))
    )
    best_model_path = os.path.join(load_dir, "best_model.pt")


    train_losses = []
    val_losses = []

    if not (os.path.exists(best_model_path) or checkpoint_files):
        print("No checkpoints found, training from scratch...")    
    else:
        if checkpoint_files:
            latest_ckpt = checkpoint_files[-1]
        elif os.path.exists(best_model_path):
            print(f"Found best model {best_model_path}, resuming training...")
            latest_ckpt = best_model_path # if no checkpoint, fallback to resuming with best model
        print(f"Found checkpoint {latest_ckpt}, resuming training...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        vicGPT.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["last_completed_epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        best_val_loss = min(val_losses) if val_losses else float("inf")
        print(f"Resuming from epoch {start_epoch+1}, best validation loss so far: {best_val_loss}")

    #vicGPT.to(device)
    print(f"Using device {device}")
    
    #TODO
    
    #TODO UNCOMMENT WHEN DONE TESTING
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
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        vicGPT.train()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch+1}")): 
            input_ids = batch["input_ids"].to(device) # (batch_size, seq_len)
            attention_mask = batch["attention_mask"].to(device) # (batch_size, seq_len)
            #print(f"batch id {batch_idx}")
            #print("Batch input_ids:")
            #print(input_ids)
            #print("Batch attention mask:")
            #print(attention_mask)
            #print(f"input ids shape: {input_ids.shape}") 
            #####################
            loss = training_loss(vicGPT, input_ids, attention_mask, tokenizer.vocab_size)
            #print(f"loss: {loss}")
            ####################### 
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss/len(train_loader)
        train_losses.append(avg_epoch_loss)
        
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
                loss = training_loss(vicGPT, input_ids, attention_mask, tokenizer.vocab_size)
                #######################
                val_loss += loss.item()
        avg_epoch_loss = val_loss/len(val_loader)
        val_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Train loss: {train_losses[-1]} Validation Loss: {val_losses[-1]}") 
        run.log({"train_loss": train_losses[-1], "val_loss":val_losses[-1]})
        
        # === Save checkpoint for this epoch ===
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
        save_dict = {
            "last_completed_epoch": epoch,
            "model_state_dict": vicGPT.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "number_training_tokens": train_total_tokens,
        }
        torch.save(save_dict,ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
        if val_losses[-1] < best_val_loss:
            # === Save best model ===
            best_val_loss = val_losses[-1]
            best_model_path = os.path.join(save_dir, "best_model.pt")
            torch.save(save_dict, best_model_path)
            print(f"Validation loss decreased, saved at {best_model_path}")


    print("-"*50)
    print("Done training")
    run.finish()
    # plot losses
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()
    print(train_losses)
    print(val_losses)
    print("Done!")

def get_args():
    parser = argparse.ArgumentParser(description="Train VicGPT")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--pre_training", action="store_true",
                            help="Run pre-training mode")
    mode_group.add_argument("--sft", action="store_true",
                            help="Run supervised fine-tuning mode")

    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--name_output_model", type=str, default="vicGPT_model")

    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_proc", type=int, default=8)

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=32)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(args)
    exit()
