#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import gc

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import torch
from transformers import T5EncoderModel, T5Tokenizer
import pathlib

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, PixArtAlphaPipeline, Transformer2DModel
from scripts.diffusers_patches import pixart_sigma_init_patched_inputs

interpolation_scale_sigma = {256: 0.5, 512: 1, 1024: 2, 2048: 4}

def main(args):
    # load the first checkpoint
    repo_path_a = args.repo_path_a
    repo_path_b = args.repo_path_b
    output_folder = pathlib.Path(args.output_folder)
    ratio = args.ratio

    setattr(Transformer2DModel, '_init_patched_inputs', pixart_sigma_init_patched_inputs)
    transformer_a = Transformer2DModel.from_pretrained(repo_path_a, subfolder='transformer')
    state_dict_a = transformer_a.state_dict()

    # load the second checkpoint
    transformer_b = Transformer2DModel.from_pretrained(repo_path_b, subfolder='transformer')
    state_dict_b = transformer_b.state_dict()

    new_state_dict = {}

    for key, value in state_dict_a.items():
        value_a = state_dict_a[key]
        value_b = state_dict_b[key]
        new_val = torch.lerp(value_a, value_b, ratio)
        new_state_dict[key] = new_val
    
    # delete the transformers to reduce RAM requirements
    del transformer_a
    del transformer_b
    del state_dict_a
    del state_dict_b
    gc.collect()

    # save the new transformer
    new_transformer = Transformer2DModel.from_pretrained(repo_path_a, subfolder='transformer')
    new_transformer.load_state_dict(new_state_dict)
    new_transformer.save_pretrained(output_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path_a', required=True, type=str)
    parser.add_argument('--repo_path_b', required=True, type=str)
    parser.add_argument('--output_folder', required=True, type=str)
    parser.add_argument('--ratio', required=True, type=float)
    parser.add_argument('--version', required=False, default='sigma', type=str)

    args = parser.parse_args()
    main(args)