import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import logging
import math
import os
from pathlib import Path
import datasets
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, AutoencoderTiny
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.pil_utils import numpy_to_pil
from diffusers.training_utils import EMAModel
from transformers import T5Tokenizer, T5EncoderModel
from torch.utils.data import Dataset

from diffusion.data.builder import build_dataset, build_dataloader, set_data_root
from diffusion.data.transforms import get_transform
from scripts.DMD.transformer_train.utils import accelerate_save_state
from diffusion.data.datasets import DMD
from diffusion.utils.misc import read_config
from diffusers import Transformer2DModel
from diffusion.utils.dist_utils import flush
from scripts.DMD.transformer_train.args import parse_args
from scripts.DMD.transformer_train.utils import save_image
from scripts.DMD.transformer_train.attention_processor import AttentionPorcessorFP32
from scripts.DMD.transformer_train.generate import generate_sample_1step, forward_model
from scripts.DMD.transformer_train.utils import compute_snr


def onestep_sampler(unet, noise_scheduler, fix_t, noise, encoder_hidden_states, uncond_encoder_hidden_states=None):
    noise_offset = torch.randn_like(noise) * 0.0
    noise_offset = noise_offset.half()

    bsz = noise.shape[0]
    timesteps = torch.zeros((bsz,), device=noise.device) + fix_t
    timesteps = timesteps.long().to(noise.device)
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noise.device, dtype=noise.dtype)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(noise.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(noise.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noise = sqrt_alpha_prod.half() * noise_offset + sqrt_one_minus_alpha_prod.half() * noise.half()

    if uncond_encoder_hidden_states:
        noise_cat = torch.cat([noise, noise], 0)
        embedding_cat = torch.cat([uncond_encoder_hidden_states.half(), encoder_hidden_states.half()], 0)

        noise_pred_uncond, noise_pred_text = (unet(noise_cat, timesteps, embedding_cat).sample).chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
    else:
        noise_pred = unet(noise, timesteps, encoder_hidden_states).sample

    pred_x0 = 1 / sqrt_alpha_prod.half() * (noise - 1.0 * sqrt_one_minus_alpha_prod.half() * noise_pred)

    return pred_x0


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {base_model}
    tags:
    - pixart
    - pixart-diffusers
    - text-to-image
    - diffusers
    inference: true
    ---
    """
    model_card = f"""
    These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. 
    You can find some example images in the following. \n {img_str}
    """
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def main():
    args = parse_args()
    config = read_config(args.config)
    torch.hub.set_dir(args.torch_hub_path)
    if args.use_dm:
        args.output_dir += '_dm'
    if args.use_regression:
        args.output_dir += '_regression'
    args.output_dir += ('_' + '{}distep'.format(args.di_steps))
    args.output_dir += (
            '_' + '{}{}sgmul{}warmup{}'.format(args.lr_scheduler, args.learning_rate, args.lr_fake_multiplier,
                                               args.lr_warmup_steps)
    )
    args.output_dir += ('_' + 'cfg{}'.format(args.cfg))
    if args.fix_noise_ts is not None:
        args.output_dir += '_{}ts{}'.format(args.start_ts, args.fix_noise_ts)
    else:
        args.output_dir += '_{}ts'.format(args.start_ts)
    if args.local_debugging:
        args.output_dir += '_debugging'
    args.output_dir += ('_' + 'acc{}'.format(args.gradient_accumulation_steps))

    args.output_dir += '_maxgrad{}_mixedprecision{}_bs{}_one_step_maxt{}'.format(args.max_grad_norm,
                                                                                 args.mixed_precision,
                                                                                 args.train_batch_size,
                                                                                 args.one_step_maxt)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logger.info(f"Config: \n{config.pretty_text}")
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    tokenizer = text_encoder = None
    if not config.data.load_t5_feat:
        tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

    logger.info(f"vae scale factor: {config.scale_factor}")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(config.load_from, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(config.vae_pretrained).to(accelerator.device)
    config.scale_factor = vae.config.scaling_factor

    ## Initialize the network architecture
    model = Transformer2DModel.from_pretrained(config.load_from, subfolder='transformer')
    model_real = Transformer2DModel.from_pretrained(config.load_from, subfolder='transformer')
    model_fake = Transformer2DModel.from_pretrained(config.load_from, subfolder='transformer')
    if args.mixed_precision == "fp16":
        for net in [model_fake, model]:
            for m in net.modules():
                if not hasattr(m, 'processor'): continue
                m.processor = AttentionPorcessorFP32()
                logger.info("replace attention with fp32 attention")

    # freeze parameters of models
    model_real.requires_grad_(False)
    vae.requires_grad_(False)

    model = accelerator.prepare(model)
    if config.grad_checkpointing:
        model.enable_gradient_checkpointing()

    model_fake = accelerator.prepare(model_fake)
    if config.grad_checkpointing:
        model_fake.enable_gradient_checkpointing()

    if args.use_ema:
        ema_model_config = Transformer2DModel.load_config(config.load_from, subfolder='transformer')
        ema_model = Transformer2DModel.from_config(ema_model_config)
        ema_model = EMAModel(model.parameters(), model_cls=Transformer2DModel, model_config=ema_model.config)

    # For mixed precision training we cast and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    model_real.to(accelerator.device, dtype=weight_dtype)

    ### Initilize VAE
    vae.to(dtype=weight_dtype)
    vae_for_regression = AutoencoderTiny.from_pretrained(config.tiny_vae_pretrained)
    vae_for_regression.requires_grad_(False)
    vae_for_regression.to(accelerator.device, dtype=weight_dtype)

    # /cache/torch/hub/checkpoints/vgg16-397923af.pth
    if args.lpips_layer != 0:
        print("use self-defined lpips", args.lpips_layer)
        lpips_loss_fn = lpips.LPIPS(args.lpips_layer, net='vgg').to(accelerator.device)
    else:
        lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
            model_fake.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    logger.info(f'adam_beta1 {args.adam_beta1}')
    optimizer = optimizer_cls(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_fake = optimizer_cls(
        model_fake.parameters(),
        lr=args.learning_rate * args.lr_fake_multiplier,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).
    set_data_root(config.data_root)
    transform = config.data.pop('transform', 'default_train')
    transform = get_transform(transform, config.image_size)
    train_dataset = DMD(root=config.data.root, resolution=config.image_size, transform=transform,
                        image_list_json=config.data.image_list_json, max_samples=args.max_samples,
                        max_length=config.model_max_length,
                        load_vae_feat=config.data.load_vae_feat, load_t5_feat=config.data.load_t5_feat, )
    test_dataset = DMD(root=config.data.root, resolution=config.image_size, transform=transform,
                       image_list_json=config.data.image_list_json, max_samples=32,
                       max_length=config.model_max_length,
                       load_vae_feat=config.data.load_vae_feat, load_t5_feat=config.data.load_t5_feat,)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler_fake = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_fake,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # accelerator._optimizers[0] : optimizer
    # accelerator._optimizers[1] : optimizer_fake
    # Prepare everything with our `accelerator`.

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    optimizer_fake, lr_scheduler_fake = accelerator.prepare(optimizer_fake, lr_scheduler_fake)
    test_dataloader = accelerator.prepare(test_dataloader)

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    # first_epoch = 0
    first_epoch = -1

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, '_args.txt'), 'w') as f:
            for k, v in args.__dict__.items():
                f.write('{}: {}\n'.format(k, v))
            f.write('\n')
            f.write('num examples: {}\n'.format(len(train_dataset)))
            f.write('total batch size: {}\n\n'.format(total_batch_size))

    # Potentially load in the weights and states from a previous save
    resume_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    save_image_steps = [1]
    image_save_path = os.path.join(args.output_dir, '_images')
    os.makedirs(image_save_path, exist_ok=True)

    write_captions = True
    test_init_noise = torch.randn((32, 4, 64, 64)).to(accelerator.device).to(weight_dtype)
    max_length = config.model_max_length

    ### Check and load previously saved uncondition text feature and mask
    if not os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token_fp32.pth'):
        if text_encoder is None or tokenizer is None:
            logger.info(f"Loading text encoder and tokenizer from {args.pipeline_load_from} ...")
            tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
            text_encoder = T5EncoderModel.from_pretrained(
                args.pipeline_load_from, subfolder="text_encoder", torch_dtype=torch.float32).to(accelerator.device)
            null_tokens = tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            torch.save(
                {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
                f'output/pretrained_models/null_embed_diffusers_{max_length}token_fp32.pth')
            if config.data.load_t5_feat:
                del tokenizer
                del text_encoder
            flush()

    negative_prompt_embeds_dict = torch.load(
        f'output/pretrained_models/null_embed_diffusers_{max_length}token_fp32.pth', map_location='cpu')
    negative_prompt_embeds = negative_prompt_embeds_dict['uncond_prompt_embeds']
    negative_prompt_attention_masks = negative_prompt_embeds_dict['uncond_prompt_embeds_mask']

    for epoch in range(first_epoch, args.num_train_epochs):

        sg_train_loss = 0.0
        g_train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            model_fake.train()
            model.train()
            accumulate_context = accelerator.accumulate(model, model_fake)

            with accumulate_context:
                y = batch['txt_fea'].squeeze(1).to(weight_dtype)
                y_mask = batch['attention_mask'].squeeze(1).squeeze(1).to(weight_dtype)

                init_noise = batch['noise'].to(weight_dtype) if args.use_regression else torch.randn(
                    (y.shape[0], 4, 64, 64), dtype=weight_dtype, device=accelerator.device)

                # generate result for one-step output of student model
                latents = generate_sample_1step(model, noise_scheduler, init_noise, args.one_step_maxt, y, y_mask)

                ################ train student model ############################
                loss = 0.0

                if args.use_dm:
                    # generate noise
                    noise = torch.randn_like(latents)
                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                        )
                    bsz = latents.shape[0]

                    # keep the time-stamps the same
                    # larger timestep will cause worse performance
                    # refer to: PixArt-Sigma paper Supplementary report: https://arxiv.org/abs/2403.04692
                    maxt = args.start_ts
                    timesteps = torch.randint(1, maxt, (bsz,), device=latents.device)
                    if args.fix_noise_ts is not None:
                        timesteps = 0 * timesteps + args.fix_noise_ts
                    timesteps = timesteps.long()
                    # add noise to the one-step result
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the target for loss depending on the prediction type
                    if args.prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                    ### Distribution matching Loss computation
                    with torch.no_grad():
                        noisy_latents_cat = torch.cat([noisy_latents, noisy_latents], 0).to(weight_dtype)
                        timesteps_cat = torch.cat([timesteps, timesteps], 0)

                        uncond_encoder_hidden_states = negative_prompt_embeds.repeat(
                            init_noise.shape[0], 1, 1).to(weight_dtype).to(init_noise.device)
                        uncond_attention_mask = negative_prompt_attention_masks.repeat(
                            init_noise.shape[0], 1).to(weight_dtype).to(init_noise.device)

                        encoder_cat = torch.cat([uncond_encoder_hidden_states, y], dim=0)
                        mask_cat = torch.cat([uncond_attention_mask, y_mask], dim=0)

                        # Real model forward
                        model_real_output = forward_model(model_real,
                                                          noisy_latents_cat,
                                                          timesteps_cat,
                                                          encoder_cat,
                                                          mask_cat)
                        score_real_uncond, score_real_cond = (-model_real_output).chunk(2)
                        score_real = score_real_uncond + args.cfg * (score_real_cond - score_real_uncond)

                        # Fake model forward
                        model_fake_output = forward_model(model_fake,
                                                          noisy_latents,
                                                          timesteps,
                                                          y,
                                                          y_mask)
                        score_fake = -model_fake_output

                        alpha_prod_t = noise_scheduler.alphas_cumprod.to(device=latents.device, dtype=latents.dtype)[timesteps]
                        beta_prod_t = 1.0 - alpha_prod_t

                        coeff = (score_fake - score_real) * beta_prod_t.view(-1, 1, 1, 1) ** 0.5 / alpha_prod_t.view(-1, 1, 1, 1) ** 0.5

                    if args.snr_gamma is None:
                        pred_latents = (
                                (
                                        noisy_latents + beta_prod_t.view(-1, 1, 1, 1) ** 0.5 * score_real
                                ) / alpha_prod_t.view(-1, 1, 1, 1) ** 0.5
                        )
                        weight = 1. / ((latents - pred_latents).abs().mean([1, 2, 3], keepdim=True) + 1e-5).detach()
                        dm_loss = F.mse_loss(latents, (latents - weight * coeff).detach())
                    else:
                        snr = compute_snr(timesteps, noise_scheduler)
                        mse_loss_weights = (
                                torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                        )
                        dm_loss = (coeff * latents).mean([1, 2, 3])
                        dm_loss = dm_loss * mse_loss_weights
                        dm_loss = dm_loss.mean()

                    loss += dm_loss

                if args.use_regression:
                    if args.use_dm:
                        regression_weight = args.regression_weight
                        regression_bsz = max(int(latents.shape[0] / 2), 1)
                        imgs = vae_for_regression.decode(latents[:regression_bsz]).sample
                        base_imgs = vae_for_regression.decode(
                            batch['base_latent'][:regression_bsz].to(weight_dtype)).sample
                    else:
                        regression_weight = 1.0
                        imgs = vae_for_regression.decode(latents).sample
                        base_imgs = vae_for_regression.decode(batch['base_latent'].to(weight_dtype)).sample
                    imgs = torch.clamp(imgs, min=-1.0, max=1.0)
                    base_imgs = torch.clamp(base_imgs, min=-1.0, max=1.0)
                    regression_loss = lpips_loss_fn(imgs, base_imgs).mean()
                    regression_loss = regression_loss * regression_weight
                    loss += regression_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                g_train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    accelerator.unscale_gradients()
                    norm_type = 2
                    params_to_clip = model.parameters()
                    if accelerator.distributed_type == DistributedType.FSDP:
                        accelerator._models[0].clip_grad_norm_(args.max_grad_norm, norm_type)
                    elif accelerator.distributed_type != DistributedType.DEEPSPEED:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm, norm_type=norm_type)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    accelerator.log({"g_train_loss": g_train_loss}, step=global_step)
                    accelerator.log({"latents": latents.abs().mean().item()}, step=global_step)
                    if args.use_dm:
                        accelerator.log({"dm_weight": weight.abs().mean().item()}, step=global_step)
                        accelerator.log({"dm_coeff": coeff.abs().mean().item()}, step=global_step)
                        accelerator.log({"dm_coeffxlatents": (coeff * latents).mean([1, 2, 3]).mean().item()},
                                        step=global_step)
                        accelerator.log({"dm_loss": dm_loss.item()}, step=global_step)
                        accelerator.log({"alpha_prod_t_sqrt": (alpha_prod_t[0] ** (0.5)).item()}, step=global_step)
                        accelerator.log({"dmd_total_loss": loss.item()}, step=global_step)
                    if args.use_regression:
                        loss_name = "lpips_loss"
                        accelerator.log({loss_name: regression_loss}, step=global_step)

                    g_train_loss = 0.0

                ################ train model_fake ################
                latents_for_fake = latents.detach()
                noise_for_fake = torch.randn_like(latents_for_fake)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise_for_fake += args.noise_offset * torch.randn(
                        (latents_for_fake.shape[0], latents_for_fake.shape[1], 1, 1), device=latents_for_fake.device)

                bsz = latents_for_fake.shape[0]
                timesteps_for_fake = torch.randint(1, args.start_ts, (bsz,), device=latents_for_fake.device)
                if args.fix_noise_ts is not None:
                    timesteps_for_fake = 0 * timesteps_for_fake + args.fix_noise_ts
                timesteps_for_fake = timesteps_for_fake.long()

                # it works when use this, but do not know why
                # noisy_latents_for_fake = scheduler.base_scheduler.add_noise(latents_for_fake, noise_for_fake, timesteps_for_fake)
                noisy_latents_for_fake = noise_scheduler.add_noise(latents_for_fake, noise_for_fake, timesteps_for_fake)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                target = noise_for_fake

                # Predict the noise residual and compute loss
                model_pred = forward_model(model_fake,
                                           noisy_latents_for_fake,
                                           timesteps_for_fake,
                                           y,
                                           y_mask)
                if args.snr_gamma is None:
                    sgloss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps, noise_scheduler)
                    mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    sgloss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    sgloss = sgloss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    sgloss = sgloss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(sgloss.repeat(args.train_batch_size)).mean()
                sg_train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(sgloss)
                if accelerator.sync_gradients:
                    accelerator.unscale_gradients()
                    norm_type = 2
                    params_to_clip = model_fake.parameters()
                    if accelerator.distributed_type == DistributedType.FSDP:
                        accelerator._models[1].clip_grad_norm_(args.max_grad_norm, norm_type)
                    elif accelerator.distributed_type != DistributedType.DEEPSPEED:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm, norm_type=norm_type)
                optimizer_fake.step()
                lr_scheduler_fake.step()
                optimizer_fake.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    accelerator.log({"sg_train_loss": sg_train_loss}, step=global_step)
                    sg_train_loss = 0.0

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    with torch.cuda.device(accelerator.device):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir,
                            f"checkpoint-{global_step}" if args.node_id == 0 else f"checkpoint-{global_step}-{args.node_id}"
                        )

                        accelerate_save_state(accelerator, save_path, save_unet_only=args.save_unet_only)
                        model.save_config(save_path)
                        logger.info(f"Saved state to {save_path}")

                        if args.use_ema:
                            ema_model.store(model.parameters())
                            ema_model.copy_to(model.parameters())
                            save_path = os.path.join(
                                args.output_dir,
                                f"ema-checkpoint-{global_step}" if args.node_id == 0 else f"ema-checkpoint-{global_step}-{args.node_id}"
                            )
                            accelerate_save_state(accelerator, save_path, save_unet_only=args.save_unet_only)
                            model.save_config(save_path)
                            logger.info(f"Saved state to {save_path}")
                            ema_model.restore(model.parameters())

                if global_step in save_image_steps or global_step % args.save_image_interval == 0:
                    curr_image_path = os.path.join(image_save_path, 'step_{:08d}'.format(global_step))
                    os.makedirs(curr_image_path, exist_ok=True)
                    for test_step, test_batch in enumerate(test_dataloader):
                        test_index = test_step + accelerator.process_index * len(test_dataloader)
                        if write_captions:
                            with open(os.path.join(image_save_path, '_captions_{}.txt'.format(args.node_id)), 'a') as f:
                                f.write('{:04d}: {}\n'.format(test_index, test_batch['text'][0]))
                        with torch.no_grad():
                            y = test_batch['txt_fea'].squeeze(1).to(weight_dtype)
                            y_mask = test_batch['attention_mask'].squeeze(1).squeeze(1).to(weight_dtype)
                            load_info = test_batch['data_info']
                            image_gt = test_batch['img_gt']

                            init_noise = test_batch['noise'].to(weight_dtype) if args.use_regression else test_init_noise[test_index: test_index + 1, :, :, :]

                            latents = generate_sample_1step(model, noise_scheduler, init_noise, args.one_step_maxt, y, y_mask)
                            _image = latents.detach() / vae.config.scaling_factor
                            image = vae.decode(_image).sample
                            save_image(image, os.path.join(curr_image_path, '{:08d}_{:04d}_output_gen.jpg'.format(global_step, test_index)))
                            save_image(image_gt, os.path.join(curr_image_path, '{:08d}_{:04d}_gt.jpg'.format(global_step, test_index)))
                            if args.use_regression:
                                reg_gen_imgs = vae_for_regression.decode(latents).sample
                                base_imgs = vae_for_regression.decode(test_batch['base_latent'].to(weight_dtype)).sample

                                save_image(reg_gen_imgs, os.path.join(curr_image_path, '{:08d}_{:04d}_reg_gen.jpg'.format(global_step, test_index)))
                                save_image(base_imgs, os.path.join(curr_image_path, '{:08d}_{:04d}_reg_base.jpg'.format(global_step, test_index)))
                        write_captions = False

            logs = {}
            logs['step_g_loss'] = loss.detach().item()
            logs['lr'] = lr_scheduler.get_last_lr()[0]
            logs['step_fakeloss'] = sgloss.detach().item()
            if args.use_dm:
                logs['step_dmloss'] = dm_loss.detach().item()
            if args.use_regression:
                logs['step_lpips_loss'] = regression_loss.detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # run inference
                images = []
                for _ in range(1):
                    _image = latents.detach()
                    _image = 1 / vae.config.scaling_factor * _image
                    image = vae.decode(_image, return_dict=False)[0]
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                    image = numpy_to_pil(image)[0]
                    images.append(image)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = model.to(torch.float32)
        save_path = os.path.join(
            args.output_dir,
            f"checkpoint-{global_step}" if args.node_id == 0 else f"checkpoint-{global_step}-{args.node_id}"
        )

        accelerate_save_state(accelerator, save_path, save_unet_only=args.save_unet_only)
        model.save_config(save_path)
        logger.info(f"Final Saved state to {save_path}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=config.load_from,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()