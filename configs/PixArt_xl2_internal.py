data_root = '/data/data'
data = dict(type='InternalData', root='images', image_list_json=['data_info.json'], transform='default_train', load_vae_feat=True, load_t5_feat=True)
image_size = 256  # the generated image resolution
train_batch_size = 32
eval_batch_size = 16
use_fsdp=False   # if use FSDP mode
valid_num=0      # take as valid aspect-ratio when sample number >= valid_num
fp32_attention = True
# model setting
model = 'PixArt_XL_2'
aspect_ratio_type = None         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = False     # if use multiscale dataset model training
pe_interpolation = 1.0    # positional embedding interpolation
# qk norm
qk_norm = False
# kv token compression
kv_compress = False
kv_compress_config = {
    'sampling': None,
    'scale_factor': 1,
    'kv_compress_layer': [],
}

# training setting
num_workers=4
train_sampling_steps = 1000
visualize=False
# Keep the same seed during validation
deterministic_validation = False
eval_sampling_steps = 250
model_max_length = 120
lora_rank = 4
num_epochs = 80
gradient_accumulation_steps = 1
grad_checkpointing = False
gradient_clip = 1.0
gc_step = 1
auto_lr = dict(rule='sqrt')
validation_prompts = [
    "dog",
    "portrait photo of a girl, photograph, highly detailed face, depth of field",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

# we use different weight decay with the official implementation since it results better result
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=3e-2, eps=1e-10)
lr_schedule = 'constant'
lr_schedule_args = dict(num_warmup_steps=500)

save_image_epochs = 1
save_model_epochs = 1
save_model_steps=1000000

sample_posterior = True
mixed_precision = 'fp16'
scale_factor = 0.18215  # ldm vae: 0.18215; sdxl vae: 0.13025
ema_rate = 0.9999
tensorboard_mox_interval = 50
log_interval = 50
cfg_scale = 4
mask_type='null'
num_group_tokens=0
mask_loss_coef=0.
load_mask_index=False    # load prepared mask_type index
# load model settings
vae_pretrained = "/cache/pretrained_models/sd-vae-ft-ema"
load_from = None
resume_from = dict(checkpoint=None, load_ema=False, resume_optimizer=True, resume_lr_scheduler=True)
snr_loss=False
real_prompt_ratio = 1.0
# classifier free guidance
class_dropout_prob = 0.1
# work dir settings
work_dir = '/cache/exps/'
s3_work_dir = None
micro_condition = False
seed = 43
skip_step=0

# LCM
loss_type = 'huber'
huber_c = 0.001
num_ddim_timesteps=50
w_max = 15.0
w_min = 3.0
ema_decay = 0.95

