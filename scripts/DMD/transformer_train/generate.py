# contains functions of generating samples
import torch
from accelerate.utils.other import extract_model_from_parallel


def model_forward(generator, encoder_hidden_states, encoder_attention_mask, added_cond_kwargs, noise, start_ts):
    if isinstance(start_ts, int):
        # convert int to long
        start_ts_net_in = torch.zeros((noise.size()[0],)) + start_ts
        start_ts_net_in = start_ts_net_in.long().to(noise.device)
    else:
        start_ts_net_in = start_ts.to(noise.device)
    noise_pred = generator(hidden_states=noise, encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask, added_cond_kwargs=added_cond_kwargs,
                           imestep=start_ts_net_in).sample
    B, C = noise.shape[:2]
    assert noise_pred.shape == (B, C * 2, *noise.shape[2:])
    noise_pred = torch.split(noise_pred, C, dim=1)[0]
    return noise_pred


def generate_sample_1step(model, scheduler, latents, maxt, prompt_embeds, prompt_attention_masks=None):
    t = torch.full((1,), maxt, device=latents.device).long()
    noise_pred = forward_model(
        model,
        latents=latents,
        timestep=t,
        prompt_embeds=prompt_embeds,
        prompt_attention_masks=prompt_attention_masks,
    )
    latents = eps_to_mu(scheduler, noise_pred, latents, t)
    return latents

def eps_to_mu(scheduler, model_output, sample, timesteps):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha_prod_t = alphas_cumprod[timesteps]
    while len(alpha_prod_t.shape) < len(sample.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    return pred_original_sample


def forward_model(model, latents, timestep, prompt_embeds, prompt_attention_masks=None):
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if extract_model_from_parallel(model).config.sample_size == 128:
        batch_size, _, height, width = latents.shape
        resolution = torch.tensor([height, width]).repeat(batch_size, 1)
        aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
        resolution = resolution.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    timestep = timestep.expand(latents.shape[0])

    noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_masks,
        added_cond_kwargs=added_cond_kwargs,
    ).sample

    if extract_model_from_parallel(model).config.out_channels // 2 == latents.shape[1]:
        noise_pred = noise_pred.chunk(2, dim=1)[0]

    return noise_pred