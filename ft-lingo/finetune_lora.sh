python finetune.py \
    --use_lora True \
    --data_path ./data/lyrics.json \
    --output_dir ./out/ \
    --per_device_train_batch_size 2 \
    --model_max_length 512 \
    --logging_steps 1 