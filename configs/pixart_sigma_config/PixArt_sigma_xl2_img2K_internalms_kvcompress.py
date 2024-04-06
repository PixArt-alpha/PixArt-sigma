_base_ = ['../PixArt_xl2_internal.py']
data_root = 'data'
image_list_json = ['data_info.json']

data = dict(
    type='InternalDataMSSigma', root='InternData', image_list_json=image_list_json, transform='default_train',
    load_vae_feat=False, load_t5_feat=False
)
image_size = 2048

# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'fp16'
fp32_attention = True
load_from = None
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_2048'  # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 4.0

# training setting
num_workers = 10
train_batch_size = 4  # 48
num_epochs = 10  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=2e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=100)

eval_sampling_steps = 100
visualize = True
log_interval = 10
save_model_epochs = 10
save_model_steps = 100
work_dir = 'output/debug'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.1
kv_compress = False
kv_compress_config = {
    'sampling': 'conv',
    'scale_factor': 2,
    'kv_compress_layer': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
}
