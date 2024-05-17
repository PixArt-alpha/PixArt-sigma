#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline, Transformer2DModel
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs


ckpt_id = "PixArt-alpha"
# https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/scripts/inference.py#L125
interpolation_scale_alpha = {256: 1, 512: 1, 1024: 2}
interpolation_scale_sigma = {256: 0.5, 512: 1, 1024: 2, 2048: 4}

def main(args):
    # load the pipe, but only the transformer
    repo_path = args.repo_path
    output_path = args.output_path
    
    setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)
    pipe = PixArtAlphaPipeline.from_pretrained(repo_path, text_encoder=None)

    transformer = pipe.transformer
    state_dict = transformer.state_dict()
    converted_state_dict = {}

    converted_state_dict['x_embedder.proj.weight'] = state_dict.pop('pos_embed.proj.weight')
    converted_state_dict['x_embedder.proj.bias'] = state_dict.pop('pos_embed.proj.bias')

    converted_state_dict['y_embedder.y_proj.fc1.weight'] = state_dict.pop('caption_projection.linear_1.weight')
    converted_state_dict['y_embedder.y_proj.fc1.bias'] = state_dict.pop('caption_projection.linear_1.bias')
    converted_state_dict['y_embedder.y_proj.fc2.weight'] = state_dict.pop('caption_projection.linear_2.weight')
    converted_state_dict['y_embedder.y_proj.fc2.bias'] = state_dict.pop('caption_projection.linear_2.bias')

    converted_state_dict['t_embedder.mlp.0.weight'] = state_dict.pop('adaln_single.emb.timestep_embedder.linear_1.weight')
    converted_state_dict['t_embedder.mlp.0.bias'] = state_dict.pop('adaln_single.emb.timestep_embedder.linear_1.bias')
    converted_state_dict['t_embedder.mlp.2.weight'] = state_dict.pop('adaln_single.emb.timestep_embedder.linear_2.weight')
    converted_state_dict['t_embedder.mlp.2.bias'] = state_dict.pop('adaln_single.emb.timestep_embedder.linear_2.bias')

    # check for micro condition??
    if 'adaln_single.emb.resolution_embedder.linear_1.weight' in state_dict:
        converted_state_dict['csize_embedder.mlp.0.weight'] = state_dict.pop('adaln_single.emb.resolution_embedder.linear_1.weight')
        converted_state_dict['csize_embedder.mlp.0.bias'] = state_dict.pop('adaln_single.emb.resolution_embedder.linear_1.bias')
        converted_state_dict['csize_embedder.mlp.2.weight'] = state_dict.pop('adaln_single.emb.resolution_embedder.linear_2.weight')
        converted_state_dict['csize_embedder.mlp.2.bias'] = state_dict.pop('adaln_single.emb.resolution_embedder.linear_2.bias')
        converted_state_dict['ar_embedder.mlp.0.weight'] = state_dict.pop('adaln_single.emb.aspect_ratio_embedder.linear_1.weight')
        converted_state_dict['ar_embedder.mlp.0.bias'] = state_dict.pop('adaln_single.emb.aspect_ratio_embedder.linear_1.bias')
        converted_state_dict['ar_embedder.mlp.2.weight'] = state_dict.pop('adaln_single.emb.aspect_ratio_embedder.linear_2.weight')
        converted_state_dict['ar_embedder.mlp.2.bias'] = state_dict.pop('adaln_single.emb.aspect_ratio_embedder.linear_2.bias')
    
    # shared norm
    converted_state_dict['t_block.1.weight'] = state_dict.pop('adaln_single.linear.weight')
    converted_state_dict['t_block.1.bias'] = state_dict.pop('adaln_single.linear.bias')

    for depth in range(28):
        converted_state_dict[f"blocks.{depth}.scale_shift_table"] = state_dict.pop(f"transformer_blocks.{depth}.scale_shift_table")

        # self attention
        q = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_q.weight')
        q_bias = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_q.bias')
        k = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_k.weight')
        k_bias = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_k.bias')
        v = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_v.weight')
        v_bias = state_dict.pop(f'transformer_blocks.{depth}.attn1.to_v.bias')
        converted_state_dict[f'blocks.{depth}.attn.qkv.weight'] = torch.cat((q, k, v))
        converted_state_dict[f'blocks.{depth}.attn.qkv.bias'] = torch.cat((q_bias, k_bias, v_bias))

        # projection
        converted_state_dict[f"blocks.{depth}.attn.proj.weight"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.to_out.0.weight")
        converted_state_dict[f"blocks.{depth}.attn.proj.bias"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.to_out.0.bias")

        # check for qk norm
        if f'transformer_blocks.{depth}.attn1.q_norm.weight' in state_dict:
            converted_state_dict[f"blocks.{depth}.attn.q_norm.weight"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.q_norm.weight")
            converted_state_dict[f"blocks.{depth}.attn.q_norm.bias"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.q_norm.bias")
            converted_state_dict[f"blocks.{depth}.attn.k_norm.weight"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.k_norm.weight")
            converted_state_dict[f"blocks.{depth}.attn.k_norm.bias"] = state_dict.pop(f"transformer_blocks.{depth}.attn1.k_norm.bias")
        
        # feed-forward
        converted_state_dict[f"blocks.{depth}.mlp.fc1.weight"] = state_dict.pop(f"transformer_blocks.{depth}.ff.net.0.proj.weight")
        converted_state_dict[f"blocks.{depth}.mlp.fc1.bias"] = state_dict.pop(f"transformer_blocks.{depth}.ff.net.0.proj.bias")
        converted_state_dict[f"blocks.{depth}.mlp.fc2.weight"] = state_dict.pop(f"transformer_blocks.{depth}.ff.net.2.weight")
        converted_state_dict[f"blocks.{depth}.mlp.fc2.bias"] = state_dict.pop(f"transformer_blocks.{depth}.ff.net.2.bias")

        # cross-attention
        q = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_q.weight")
        q_bias = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_q.bias")
        k = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_k.weight")
        k_bias = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_k.bias")
        v = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_v.weight")
        v_bias = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_v.bias")

        converted_state_dict[f"blocks.{depth}.cross_attn.q_linear.weight"] = q
        converted_state_dict[f"blocks.{depth}.cross_attn.q_linear.bias"] = q_bias
        converted_state_dict[f"blocks.{depth}.cross_attn.kv_linear.weight"] = torch.cat((k, v))
        converted_state_dict[f"blocks.{depth}.cross_attn.kv_linear.bias"] = torch.cat((k_bias, v_bias))

        converted_state_dict[f"blocks.{depth}.cross_attn.proj.weight"] = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_out.0.weight")
        converted_state_dict[f"blocks.{depth}.cross_attn.proj.bias"] = state_dict.pop(f"transformer_blocks.{depth}.attn2.to_out.0.bias")
    
    # final block
    converted_state_dict["final_layer.linear.weight"] = state_dict.pop("proj_out.weight")
    converted_state_dict["final_layer.linear.bias"] = state_dict.pop("proj_out.bias")
    converted_state_dict["final_layer.scale_shift_table"] = state_dict.pop("scale_shift_table")

    # save the state_dict
    to_save = {}
    to_save['state_dict'] = converted_state_dict
    torch.save(to_save, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', type=str, required=True, help='Path to the diffusers folder or huggingface repository')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()
    main(args)