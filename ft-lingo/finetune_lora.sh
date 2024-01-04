python finetune.py \
    --use_lora True \
    --data_path ./data/lyrics.json \
    --train_format input-output \
    --output_dir ./out/ \
    --per_device_train_batch_size 1 \
    --model_max_length 256 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --bf16 True 