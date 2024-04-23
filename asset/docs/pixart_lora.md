## Summary

**We adapt from the LoRA training code from [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) 
to achieve Transformer-LoRA fine-tuning. This document will guide you how to train and test.**

> [!IMPORTANT]  
> Somehow due to the implementation of `diffusers` and `transformers`,
> LoRA training for `transformers` can only be done in FP32.
> 
> We welcome everyone to help for solving this issue.

## How to Train
### 🔥 Run

```bash
pip install peft==0.6.2

accelerate launch --num_processes=1 --main_process_port=36667  train_scripts/train_pixart_lora_hf.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=PixArt-alpha/PixArt-XL-2-512x512 \
  --dataset_name=lambdalabs/pokemon-blip-captions --caption_column="text" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=16 \
  --num_train_epochs=200 --checkpointing_steps=100 \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="pixart-pokemon-model" \
  --validation_prompt="cute dragon creature" \
  --report_to="tensorboard" \
  --gradient_checkpointing \
  --checkpoints_total_limit=10 \
  --validation_epochs=5 \
  --rank=16
```

## How to Test

```python
import torch
from diffusers import PixArtAlphaPipeline
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# You can replace the checkpoint id with "PixArt-alpha/PixArt-XL-2-1024-MS" too.
MODEL_ID = "PixArt-alpha/PixArt-XL-2-512x512"

# LoRA model
transformer = Transformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.float16)
transformer = PeftModel.from_pretrained(transformer, "Your-LoRA-Model-Path")

# Pipeline
pipe = PixArtAlphaPipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.float16)
del transformer

pipe.to(device)

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
image.save("./catcus.png")
```