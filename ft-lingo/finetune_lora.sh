python finetune.py \
    --use_lora True \
    --data_path ./data/lyrics.json \
    --output_dir ./out/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_max_length 512 \
    --logging_steps 1 