pip install -U peft

dataset_id=svjack/pokemon-blip-captions-en-zh
model_id=PixArt-alpha/PixArt-XL-2-512x512

accelerate launch --num_processes=1 --main_process_port=36667  train_scripts/train_pixart_lora_hf.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$model_id \
  --dataset_name=$dataset_id \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=80 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="output/pixart-pokemon-model" \
  --validation_prompt="cute dragon creature" \
  --report_to="tensorboard" \
  --gradient_checkpointing \
  --checkpoints_total_limit=10 \
  --validation_epochs=5 \
  --max_token_length=120 \
  --rank=16