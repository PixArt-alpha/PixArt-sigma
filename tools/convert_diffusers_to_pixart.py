#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from safetensors import safe_open

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch

ckpt_id = "PixArt-alpha"
# https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/scripts/inference.py#L125
interpolation_scale_alpha = {256: 1, 512: 1, 1024: 2}
interpolation_scale_sigma = {256: 0.5, 512: 1, 1024: 2, 2048: 4}

def main(args):
    # load the pipe, but only the transformer
    repo_path = args.safetensor_path
    output_path = args.pth_path

    transformer = safe_open(repo_path, framework='pt')

    state_dict = transformer.keys()

    layer_depth = sum([key.endswith("attn1.to_out.0.weight") for key in state_dict]) or 28

    # check for micro condition??
    converted_state_dict = {
        'x_embedder.proj.weight': transformer.get_tensor('pos_embed.proj.weight'),
        'x_embedder.proj.bias': transformer.get_tensor('pos_embed.proj.bias'),
        'y_embedder.y_proj.fc1.weight': transformer.get_tensor('caption_projection.linear_1.weight'),
        'y_embedder.y_proj.fc1.bias': transformer.get_tensor('caption_projection.linear_1.bias'),
        'y_embedder.y_proj.fc2.weight': transformer.get_tensor('caption_projection.linear_2.weight'),
        'y_embedder.y_proj.fc2.bias': transformer.get_tensor('caption_projection.linear_2.bias'),
        't_embedder.mlp.0.weight': transformer.get_tensor('adaln_single.emb.timestep_embedder.linear_1.weight'),
        't_embedder.mlp.0.bias': transformer.get_tensor('adaln_single.emb.timestep_embedder.linear_1.bias'),
        't_embedder.mlp.2.weight': transformer.get_tensor('adaln_single.emb.timestep_embedder.linear_2.weight'),
        't_embedder.mlp.2.bias': transformer.get_tensor('adaln_single.emb.timestep_embedder.linear_2.bias')
    }
    if 'adaln_single.emb.resolution_embedder.linear_1.weight' in state_dict:
        converted_state_dict['csize_embedder.mlp.0.weight'] = transformer.get_tensor('adaln_single.emb.resolution_embedder.linear_1.weight')
        converted_state_dict['csize_embedder.mlp.0.bias'] = transformer.get_tensor('adaln_single.emb.resolution_embedder.linear_1.bias')
        converted_state_dict['csize_embedder.mlp.2.weight'] = transformer.get_tensor('adaln_single.emb.resolution_embedder.linear_2.weight')
        converted_state_dict['csize_embedder.mlp.2.bias'] = transformer.get_tensor('adaln_single.emb.resolution_embedder.linear_2.bias')
        converted_state_dict['ar_embedder.mlp.0.weight'] = transformer.get_tensor('adaln_single.emb.aspect_ratio_embedder.linear_1.weight')
        converted_state_dict['ar_embedder.mlp.0.bias'] = transformer.get_tensor('adaln_single.emb.aspect_ratio_embedder.linear_1.bias')
        converted_state_dict['ar_embedder.mlp.2.weight'] = transformer.get_tensor('adaln_single.emb.aspect_ratio_embedder.linear_2.weight')
        converted_state_dict['ar_embedder.mlp.2.bias'] = transformer.get_tensor('adaln_single.emb.aspect_ratio_embedder.linear_2.bias')

    # shared norm
    converted_state_dict['t_block.1.weight'] = transformer.get_tensor('adaln_single.linear.weight')
    converted_state_dict['t_block.1.bias'] = transformer.get_tensor('adaln_single.linear.bias')

    for depth in range(layer_depth):
        print(f"Converting layer {depth}")
        converted_state_dict[f"blocks.{depth}.scale_shift_table"] = transformer.get_tensor(f"transformer_blocks.{depth}.scale_shift_table")

        # self attention
        q = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_q.weight')
        q_bias = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_q.bias')
        k = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_k.weight')
        k_bias = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_k.bias')
        v = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_v.weight')
        v_bias = transformer.get_tensor(f'transformer_blocks.{depth}.attn1.to_v.bias')
        converted_state_dict[f'blocks.{depth}.attn.qkv.weight'] = torch.cat((q, k, v))
        converted_state_dict[f'blocks.{depth}.attn.qkv.bias'] = torch.cat((q_bias, k_bias, v_bias))

        # projection
        converted_state_dict[f"blocks.{depth}.attn.proj.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.to_out.0.weight")
        converted_state_dict[f"blocks.{depth}.attn.proj.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.to_out.0.bias")

        # check for qk norm
        if f'transformer_blocks.{depth}.attn1.q_norm.weight' in state_dict:
            converted_state_dict[f"blocks.{depth}.attn.q_norm.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.q_norm.weight")
            converted_state_dict[f"blocks.{depth}.attn.q_norm.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.q_norm.bias")
            converted_state_dict[f"blocks.{depth}.attn.k_norm.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.k_norm.weight")
            converted_state_dict[f"blocks.{depth}.attn.k_norm.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn1.k_norm.bias")

        # feed-forward
        converted_state_dict[f"blocks.{depth}.mlp.fc1.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.ff.net.0.proj.weight")
        converted_state_dict[f"blocks.{depth}.mlp.fc1.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.ff.net.0.proj.bias")
        converted_state_dict[f"blocks.{depth}.mlp.fc2.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.ff.net.2.weight")
        converted_state_dict[f"blocks.{depth}.mlp.fc2.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.ff.net.2.bias")

        # cross-attention
        q = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_q.weight")
        q_bias = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_q.bias")
        k = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_k.weight")
        k_bias = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_k.bias")
        v = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_v.weight")
        v_bias = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_v.bias")

        converted_state_dict[f"blocks.{depth}.cross_attn.q_linear.weight"] = q
        converted_state_dict[f"blocks.{depth}.cross_attn.q_linear.bias"] = q_bias
        converted_state_dict[f"blocks.{depth}.cross_attn.kv_linear.weight"] = torch.cat((k, v))
        converted_state_dict[f"blocks.{depth}.cross_attn.kv_linear.bias"] = torch.cat((k_bias, v_bias))

        converted_state_dict[f"blocks.{depth}.cross_attn.proj.weight"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_out.0.weight")
        converted_state_dict[f"blocks.{depth}.cross_attn.proj.bias"] = transformer.get_tensor(f"transformer_blocks.{depth}.attn2.to_out.0.bias")

    # final block
    converted_state_dict["final_layer.linear.weight"] = transformer.get_tensor("proj_out.weight")
    converted_state_dict["final_layer.linear.bias"] = transformer.get_tensor("proj_out.bias")
    converted_state_dict["final_layer.scale_shift_table"] = transformer.get_tensor("scale_shift_table")

    # save the state_dict
    to_save = {}
    to_save['state_dict'] = converted_state_dict
    torch.save(to_save, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--safetensor_path', type=str, required=True, help='Path and filename of a safetensor file to convert. i.e. output/mymodel.safetensors')
    parser.add_argument('--pth_path', type=str, required=True, help='Path and filename to the output file i.e. output/mymodel.pth')

    args = parser.parse_args()
    main(args)
