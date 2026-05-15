import torch
from src.model.vic_gpt import VicGPT
from transformers import AutoTokenizer
import json

if __name__ == "__main__":
    prompt = str(input("Prompt:"))#"One day, a little girl named Lily found a needle"
    max_new_tokens = int(input("Max new tokens:"))
    # Configuration
    path = "models/pre_trained/vicGPTv2/"
    model_path = path+"best_model.pt"#vicGPT_Overfit1Example/checkpoint_epoch99.pt"
    config_path = path+"config.json"#vicGPT_Overfit1Example/config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"model trained on {torch.load(model_path, map_location=device)['number_training_tokens']} tokens") 
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config_dict["tokenizer_name"])
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    
    # Initialize model with same config as training
    vicGPT = VicGPT(
        vocab_size=tokenizer.vocab_size,
        num_layers=config_dict["num_layers"],
        embed_dim=config_dict["embed_dim"],
        num_heads=config_dict["num_heads"],
        max_seq_len=config_dict["max_seq_len"],
        mlp_hidden_dim=config_dict["mlp_hidden_dim"],
        dropout_prob=config_dict["dropout_prob"]
    )
    
    # Load weights
    vicGPT.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    vicGPT.to(device)
    vicGPT.eval()
    print(f"Model loaded from {model_path}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: '{prompt}'")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    # Generate
    print("\n" + "="*50)
    print("GENERATING...")
    print("="*50)
    output_ids = vicGPT.generate(input_ids, max_new_tokens=max_new_tokens, temperature=0.1, top_p=0.9, eos_token_id=tokenizer.eos_token_id) 
    
    output_text = tokenizer.decode(output_ids[0])
    print(output_text)
