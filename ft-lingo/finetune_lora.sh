python finetune.py \
    --use_lora True \
    --data_path ./data/train_mini.json \
    --eval_path ./data/val_mini.json \
    --output_dir ./out/ \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --model_max_length 512 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --per_device_eval_batch_size 5 \
    --num_train_epochs 50 \
    --logging_steps 1 