export HF_DATASETS_CACHE=./data
python -m src.training.training \
    --pre_training \
    --dataset_name="roneneldan/TinyStories" \
    --tokenizer_name="gpt2" \
    --name_output_model="vicGPTv2" \
    --max_seq_len=2048 \
    --num_proc=8 \
    --train_batch_size=4 \
    --val_batch_size=4 \
    --num_epochs=10 \
    --lr=2e-4 \
    --weight_decay=0.01 \
    --num_layers=3 \
    --embed_dim=512 \
    --num_heads=4 \
    --mlp_hidden_dim=1024 \
    --dropout_prob=0.1 \

