from __future__ import annotations

import os
import re
import shutil

from accelerate.checkpointing import save_accelerator_state, save_custom_state
from accelerate.logging import get_logger
from accelerate.utils import (
    MODEL_NAME,
    DistributedType,
    save_fsdp_model,
    save_fsdp_optimizer,
    is_deepspeed_available
)

from PIL import Image

if is_deepspeed_available():
    import deepspeed

    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

logger = get_logger(__name__)

############# Saving model utils

def accelerate_save_state(accelerator, output_dir=None, save_unet_only=False, unet_id=0, **save_model_func_kwargs):
    """
    Saves the current states of the model, optimizer, scaler, RNG generators, and registered objects to a folder.

    If a `ProjectConfiguration` was passed to the `Accelerator` object with `automatic_checkpoint_naming` enabled
    then checkpoints will be saved to `self.project_dir/checkpoints`. If the number of current saves is greater
    than `total_limit` then the oldest save is deleted. Each checkpoint is saved in seperate folders named
    `checkpoint_<iteration>`.

    Otherwise they are just saved to `output_dir`.

    <Tip>

    Should only be used when wanting to save a checkpoint during training and restoring the state in the same
    environment.

    </Tip>

    Args:
        output_dir (`str` or `os.PathLike`):
            The name of the folder to save all relevant weights and states.
        save_model_func_kwargs (`dict`, *optional*):
            Additional keyword arguments for saving model which can be passed to the underlying save function, such
            as optional arguments for DeepSpeed's `save_checkpoint` function.

    Example:

    ```python
    >>> from accelerate import Accelerator

    >>> accelerator = Accelerator()
    >>> model, optimizer, lr_scheduler = ...
    >>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    >>> accelerator.save_state(output_dir="my_checkpoint")
    ```
    """
    if accelerator.project_configuration.automatic_checkpoint_naming:
        output_dir = os.path.join(accelerator.project_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    if accelerator.project_configuration.automatic_checkpoint_naming:
        folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
        if accelerator.project_configuration.total_limit is not None and (
            len(folders) + 1 > accelerator.project_configuration.total_limit
        ):

            def _inner(folder):
                return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

            folders.sort(key=_inner)
            logger.warning(
                f"Deleting {len(folders) + 1 - accelerator.project_configuration.total_limit} checkpoints to make room for new checkpoint."
            )
            for folder in folders[: len(folders) + 1 - accelerator.project_configuration.total_limit]:
                shutil.rmtree(folder)
        output_dir = os.path.join(output_dir, f"checkpoint_{accelerator.save_iteration}")
        if os.path.exists(output_dir):
            raise ValueError(
                f"Checkpoint directory {output_dir} ({accelerator.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
            )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving current state to {output_dir}")


    # Save the models taking care of FSDP and DeepSpeed nuances

    weights = []
    for i, model in enumerate(accelerator._models):
        if save_unet_only and i != unet_id:
            continue
        if accelerator.distributed_type == DistributedType.FSDP:
            logger.info("Saving FSDP model")
            save_fsdp_model(accelerator.state.fsdp_plugin, accelerator, model, output_dir, i)
            logger.info(f"FSDP Model saved to output dir {output_dir}")
        elif accelerator.distributed_type == DistributedType.DEEPSPEED:
            logger.info("Saving DeepSpeed Model and Optimizer")
            ckpt_id = f"{MODEL_NAME}" if i == 0 else f"{MODEL_NAME}_{i}"
            model.save_checkpoint(output_dir, ckpt_id, **save_model_func_kwargs)
            logger.info(f"DeepSpeed Model and Optimizer saved to output dir {os.path.join(output_dir, ckpt_id)}")
        elif accelerator.distributed_type == DistributedType.MEGATRON_LM:
            logger.info("Saving Megatron-LM Model, Optimizer and Scheduler")
            model.save_checkpoint(output_dir)
            logger.info(f"Megatron-LM Model , Optimizer and Scheduler saved to output dir {output_dir}")
        else:
            weights.append(accelerator.get_state_dict(model, unwrap=False))

    # Save the optimizers taking care of FSDP and DeepSpeed nuances
    optimizers = []
    if not save_unet_only:
        if accelerator.distributed_type == DistributedType.FSDP:
            for i, opt in enumerate(accelerator._optimizers):
                logger.info("Saving FSDP Optimizer")
                save_fsdp_optimizer(accelerator.state.fsdp_plugin, accelerator, opt, accelerator._models[i], output_dir, i)
                logger.info(f"FSDP Optimizer saved to output dir {output_dir}")
        elif accelerator.distributed_type not in [DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM]:
            optimizers = accelerator._optimizers

    # Save the lr schedulers taking care of DeepSpeed nuances
    schedulers = []
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        for i, scheduler in enumerate(accelerator._schedulers):
            if isinstance(scheduler, DeepSpeedSchedulerWrapper):
                continue
            schedulers.append(scheduler)
    elif accelerator.distributed_type not in [DistributedType.MEGATRON_LM]:
        schedulers = accelerator._schedulers

    # Call model loading hooks that might have been registered with
    # accelerator.register_model_state_hook
    for hook in accelerator._save_model_state_pre_hook.values():
        hook(accelerator._models, weights, output_dir)

    save_location = save_accelerator_state(
        output_dir, weights, optimizers, schedulers, accelerator.state.process_index, accelerator.scaler
    )
    for i, obj in enumerate(accelerator._custom_objects):
        save_custom_state(obj, output_dir, i)

    accelerator.project_configuration.iteration += 1
    return save_location


#### calculation
def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def save_image(image, path):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = [Image.fromarray(im) for im in image]
    image[0].save(path)
