from diffusers import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from diffusers import PixArtSigmaPipeline, PixArtTransformer2DModel, DDPMScheduler
from transformers import T5EncoderModel
from pathlib import Path
import tqdm
from os import walk
import os
from copy import copy, deepcopy
import torch
from torch import nn
import torch.distributed as dist
import torch.distributed
from torch import optim
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, ASPECT_RATIO_256_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from datasets import load_dataset
from io import BytesIO
from tqdm.contrib.concurrent import process_map
from torchvision.transforms import Resize
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import datasets
from accelerate import Accelerator
import random
import gc
from datetime import datetime
import argparse
import requests
from datetime import timedelta
from torch.multiprocessing import set_start_method

def find_closest_resolution(ratio, resolution):
    if resolution == 512:
        ratio = min(ASPECT_RATIO_512_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_512_BIN[ratio]
    elif resolution == 256:
        ratio = min(ASPECT_RATIO_256_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_256_BIN[ratio]
    elif resolution == 1024:
        ratio = min(ASPECT_RATIO_1024_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_1024_BIN[ratio]
    else:
        ratio = min(ASPECT_RATIO_2048_BIN, key=lambda x:abs(float(x) - ratio))
        return ASPECT_RATIO_2048_BIN[ratio]

class FilesDataset(Dataset):
    def __init__(self, folder, extensions):
        self.folder = Path(folder)
        self.files = []

        for (dirpath, dirnames, filenames) in walk(self.folder):
            for filename in filenames:
                if not Path(filename).suffix in extensions:
                    continue
                self.files.append(Path(dirpath).joinpath(filename))
    
    def __len__(self):
        test = len(self.files)
        return test
    
    def __getitem__(self, idx):
        test = str(self.files[idx])
        return test

class PromptsDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]
        
def flush():
    gc.collect()
    torch.cuda.empty_cache()

def extract_t5_embeddings(repository_path : str, captions_folder : str, output_folder : str):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)

    output_folder = Path(output_folder)
    embeddings_folder = Path(output_folder).joinpath('embeddings')
    captions_folder = Path(captions_folder)
    embeddings_folder.mkdir(parents=True, exist_ok=True)

    captions_dataset = FilesDataset(captions_folder, ['.txt'])
    captions_loader = DataLoader(captions_dataset)

    accelerator = Accelerator()
    captions_loader, pipe = accelerator.prepare(captions_loader, pipe)
    pipe = pipe.to('cuda')

    for filepath in tqdm.tqdm(captions_loader):
        with open(filepath[0], encoding='utf-8') as f:
            prompt = f.read()

        with torch.no_grad():
            embeddings = pipe.encode_prompt(prompt)
        output_file_path = embeddings_folder.joinpath(Path(filepath[0]).stem + '.emb')
        torch.save(embeddings, output_file_path)
    del pipe
    flush()

def process_t5_shard(lock, process_index, shard : datasets.Dataset, repository_path, dataset_caption_column, queue : mp.Queue):
    device = f"cuda:{(process_index or 0) % torch.cuda.device_count()}"
    print(f'process_t5_shard(), device={device}')

    #with lock:
        #pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)
        #pipe = pipe.to(device=device)
        #print(f'process_t5_shard(), pipe loaded on device {device}')

    def add_t5_columns(batch, dataset_caption_column):
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)
        pipe = pipe.to(device=device)
        print(f'process_t5_shard(), pipe loaded on device {device}')
    
        l_prompt_embeds = []
        l_prompt_attention_mask = []
        l_negative_embeds = []
        l_negative_prompt_attention_mask = []
        for elem in tqdm.tqdm(batch[dataset_caption_column]):
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(elem)
            prompt_embeds = prompt_embeds.to('cpu')
            prompt_attention_mask = prompt_attention_mask.to('cpu')
            negative_embeds = negative_embeds.to('cpu')
            negative_prompt_attention_mask = negative_prompt_attention_mask.to('cpu')

            l_prompt_embeds.append(prompt_embeds)
            l_prompt_attention_mask.append(prompt_attention_mask)
            l_negative_embeds.append(negative_embeds)
            l_negative_prompt_attention_mask.append(negative_prompt_attention_mask)
            del prompt_embeds
            del prompt_attention_mask
            del negative_embeds
            del negative_prompt_attention_mask
            flush()
        
        batch['t5_prompt_embeds'] = l_prompt_embeds
        batch['t5_prompt_attention_mask'] = l_prompt_attention_mask
        batch['t5_negative_embeds'] = l_negative_embeds
        batch['t5_negative_prompt_attention_mask'] = l_negative_prompt_attention_mask
        return batch
    
    shard = shard.map(add_t5_columns, batched=True, batch_size=500, fn_kwargs={'dataset_caption_column' : dataset_caption_column})
    queue.put(shard)
    print(f'process_t5_shard(), shard put into queue on {device}')

def extract_t5_embeddings_from_dataset_old(repository_path, dataset : datasets.Dataset, dataset_caption_column, output_folder):
    lock = mp.Lock()
    queue = mp.Queue()

    # split the dataset into shards
    num_processes = torch.cuda.device_count()
    shards = [dataset.shard(num_processes, index=i) for i in range(num_processes)]

    processes = []
    for process_index in range(num_processes):
        p = mp.Process(target=process_t5_shard, args=(lock, process_index, shards[process_index], repository_path, dataset_caption_column, queue))
        p.start()
        processes.append(p)
    
    results = []
    
    for p in processes:
        results.append(queue.get())

    print('extract_t5_embeddings_from_dataset(), joining processes')
    for p in processes:
        p.join()
    queue.close()
    queue.join_thread()

    dataset = datasets.concatenate_datasets(results)
    return dataset

def extract_t5_embeddings_from_dataset(repository_path, dataset : datasets.Dataset, dataset_caption_column : str, t5_num_processes : int):
    def add_t5_columns(batch, rank, dataset_caption_column):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        print(f'process_t5_shard(), device={device}')
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)
        pipe = pipe.to(device=device)
        print(f'process_t5_shard(), pipe loaded on device {device}')
    
        l_prompt_embeds = []
        l_prompt_attention_mask = []
        l_negative_embeds = []
        l_negative_prompt_attention_mask = []
        for elem in tqdm.tqdm(batch[dataset_caption_column]):
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(elem)
            prompt_embeds = prompt_embeds.to('cpu')
            prompt_attention_mask = prompt_attention_mask.to('cpu')
            negative_embeds = negative_embeds.to('cpu')
            negative_prompt_attention_mask = negative_prompt_attention_mask.to('cpu')

            l_prompt_embeds.append(prompt_embeds)
            l_prompt_attention_mask.append(prompt_attention_mask)
            l_negative_embeds.append(negative_embeds)
            l_negative_prompt_attention_mask.append(negative_prompt_attention_mask)
            del prompt_embeds
            del prompt_attention_mask
            del negative_embeds
            del negative_prompt_attention_mask
            flush()
        
        batch['t5_prompt_embeds'] = l_prompt_embeds
        batch['t5_prompt_attention_mask'] = l_prompt_attention_mask
        batch['t5_negative_embeds'] = l_negative_embeds
        batch['t5_negative_prompt_attention_mask'] = l_negative_prompt_attention_mask
        return batch
    
    dataset = dataset.map(add_t5_columns, batched=True, with_rank=True, num_proc=torch.cuda.device_count() * t5_num_processes, fn_kwargs={'dataset_caption_column' : dataset_caption_column})
    return dataset
     

def extract_t5_validation_embeddings(repository_path : str, prompts : list[str], output_folder : str):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)

    output_folder = Path(output_folder)
    embeddings_folder = Path(output_folder).joinpath('validation_embeddings')
    embeddings_folder.mkdir(parents=True, exist_ok=True)
    pipe = pipe.to('cuda')

    for idx in range(len(prompts)):
        prompt = prompts[idx]
        with torch.no_grad():
            embeddings = pipe.encode_prompt(prompt)
        output_file_path = embeddings_folder.joinpath(f'{idx}.emb')
        torch.save(embeddings, output_file_path)
    del pipe
    flush()

def extract_vae_features(repository_path : str, images_folder : str, output_folder : str):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16
                                               ).to('cuda')
    image_processor = pipe.image_processor
    images_folder = Path(images_folder)
    output_folder = Path(output_folder)
    vae_features_folder = Path(output_folder).joinpath('features')
    vae_features_folder.mkdir(parents=True, exist_ok=True)

    interpolation_scale = pipe.transformer.config.interpolation_scale
    checkpoint_resolution = 512 * interpolation_scale

    del pipe.transformer
    vae = pipe.vae

    images_dataset = FilesDataset(images_folder, ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'])
    images_loader = DataLoader(images_dataset)

    accelerator = Accelerator()
    images_loader, pipe = accelerator.prepare(images_loader, pipe)
    pipe = pipe.to('cuda')

    for filename in tqdm.tqdm(images_loader):
        images_filepath = filename[0]

        bw_image = Image.open(images_filepath)
        image = Image.new('RGB', bw_image.size)
        image.paste(bw_image)

        image = image_processor.pil_to_numpy(image)
        image = torch.tensor(image, device='cuda', dtype=torch.float16)
        image = torch.moveaxis(image, -1, 1)

        # find the closest ratio
        ratio = image.shape[2] / image.shape[3]
        resolution = find_closest_resolution(ratio, checkpoint_resolution)

        image_width = int(resolution[1])
        image_height = int(resolution[0])

        resize_transform = Resize((image_height, image_width))
        image = resize_transform(image)

        with torch.no_grad():
            image = image_processor.preprocess(image)
            latent = vae.encode(image).latent_dist.sample()
            latent = latent * vae.config.scaling_factor
            del image
            flush()
        latent_filepath = vae_features_folder.joinpath(Path(filename[0]).stem + '.lat')
        torch.save(latent, latent_filepath)

def process_vae_shard(lock, process_index, repository_path, shard, dataset_url_column, dataset_images_column, output_folder, queue : mp.Queue):
    device = f"cuda:{(process_index or 0) % torch.cuda.device_count()}"

    with lock:
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, torch_dtype=torch.float16)
        pipe = pipe.to(device=device)
        image_processor = pipe.image_processor

        interpolation_scale = pipe.transformer.config.interpolation_scale
        checkpoint_resolution = 512 * interpolation_scale

        del pipe.transformer
        del pipe
        flush()
    
    def add_vae_column(batch):
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, text_encoder=None, tokenizer=None, torch_dtype=torch.float16
                                               ).to(device=device)
        vae = pipe.vae
    
        iterator = None
        if dataset_url_column != None:
            iterator = batch[dataset_url_column]
        else:
            iterator = batch[dataset_images_column]

        latents = []
        ratios = []
        for elem in tqdm.tqdm(iterator):
            if dataset_url_column != None:
                response = requests.get(elem)
                image = Image.open(BytesIO(response.content))
            else:
                image = elem
            # force rgb format
            image = image.convert('RGB')

            image = image_processor.pil_to_numpy(image)
            image = torch.tensor(image, device=device, dtype=torch.float16)
            image = torch.moveaxis(image, -1, 1)

            # find the closest ratio
            ratio = image.shape[2] / image.shape[3]
            resolution = find_closest_resolution(ratio, checkpoint_resolution)

            image_width = int(resolution[1])
            image_height = int(resolution[0])

            # use the actual ratio for the dataset
            ratio = float(image_height) / float(image_width)
            ratios.append(ratio)

            resize_transform = Resize((image_height, image_width))
            image = resize_transform(image)

            with torch.no_grad():
                image = image_processor.preprocess(image)
                latent = vae.encode(image).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                del image
                flush()
            latent = latent.to('cpu')
            latents.append(latent)

        batch[f'vae_{checkpoint_resolution}px'] = latents
        batch['ratio'] = ratios
        return batch
    shard = shard.map(add_vae_column, batched=True, batch_size=500)
    queue.put(shard)

def extract_vae_features_from_dataset_old(repository_path, dataset, dataset_url_column, dataset_images_column, output_folder):
    lock = mp.Lock()
    queue = mp.Queue()

    # split the dataset into shards
    num_processes = torch.cuda.device_count()
    shards = [dataset.shard(num_processes, index=i) for i in range(num_processes)]

    processes = []
    for process_index in range(num_processes):
        p = mp.Process(target=process_vae_shard, args=(lock, process_index, repository_path, shards[process_index], dataset_url_column, dataset_images_column, output_folder, queue))
        p.start()
        processes.append(p)
    
    results = []
    for p in processes:
        p.join()
    
    for p in processes:
        results.append(queue.get())
    queue.close()
    queue.join_thread()

    dataset = datasets.concatenate_datasets(results)
    return dataset

def extract_vae_features_from_dataset(repository_path, dataset, dataset_url_column, dataset_images_column, num_vae_processes_per_gpu):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, torch_dtype=torch.float16)
    image_processor = pipe.image_processor

    interpolation_scale = pipe.transformer.config.interpolation_scale
    checkpoint_resolution = 512 * interpolation_scale

    del pipe.transformer
    del pipe
    flush()
    
    def add_vae_column(batch, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, text_encoder=None, tokenizer=None, torch_dtype=torch.float16
                                               ).to(device=device)
        vae = pipe.vae
    
        iterator = None
        if dataset_url_column != None:
            iterator = batch[dataset_url_column]
        else:
            iterator = batch[dataset_images_column]

        latents = []
        ratios = []
        for elem in tqdm.tqdm(iterator):
            if dataset_url_column != None:
                response = requests.get(elem)
                image = Image.open(BytesIO(response.content))
            else:
                image = elem
            # force rgb format
            image = image.convert('RGB')

            image = image_processor.pil_to_numpy(image)
            image = torch.tensor(image, device=device, dtype=torch.float16)
            image = torch.moveaxis(image, -1, 1)

            # find the closest ratio
            ratio = image.shape[2] / image.shape[3]
            resolution = find_closest_resolution(ratio, checkpoint_resolution)

            image_width = int(resolution[1])
            image_height = int(resolution[0])

            # use the actual ratio for the dataset
            ratio = float(image_height) / float(image_width)
            ratios.append(ratio)

            resize_transform = Resize((image_height, image_width))
            image = resize_transform(image)

            with torch.no_grad():
                image = image_processor.preprocess(image)
                latent = vae.encode(image).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                del image
                flush()
            latent = latent.to('cpu')
            latents.append(latent)

        batch[f'vae_{checkpoint_resolution}px'] = latents
        batch['ratio'] = ratios
        return batch
    dataset = dataset.map(add_vae_column, batched=True, with_rank=True, num_proc=num_vae_processes_per_gpu * torch.cuda.device_count())
    return dataset

if __name__ == '__main__':
    # if on windows, try to set to gloo for distributed training
    try:
        if dist.is_available() and not dist.is_initialized():
            if os.name == 'nt':
                torch.distributed.init_process_group(backend='gloo')
            else:
                print('torch.distributed.init_process_group(backend=''nccl'')')
                # set to an extremely large timeout so it's not possible to get past the barrier
                torch.distributed.init_process_group(backend='nccl', timeout=timedelta(days=30))
    except:
        pass
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--repository_path', required=True, type=str)
    parser.add_argument('--images_folder', required=False, type=str, default=None)
    parser.add_argument('--captions_folder', required=False, type=str, default=None)
    parser.add_argument('-l', '--validation_prompts', nargs='+', required=False, default=None)
    parser.add_argument('--output_folder', required=True, type=str)
    parser.add_argument('--num_vae_processes', required=False, type=int, default=1)
    parser.add_argument('--num_t5_processes', required=False, type=int, default=1)
    parser.add_argument('--skip_t5_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--skip_vae_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataset_image_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_path', required=False, type=str, default=None)
    parser.add_argument('--dataset_url_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_caption_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_split', required=False, type=str, default='train')
    parser.add_argument('--dataset_output_repo', required=False, type=str, default=None)

    args = parser.parse_args()

    repository_path = args.repository_path
    validation_prompts = args.validation_prompts
    images_folder = args.images_folder
    dataset_path = args.dataset_path
    dataset_url_column = args.dataset_url_column
    dataset_image_column = args.dataset_image_column
    dataset_caption_column = args.dataset_caption_column
    dataset_split = args.dataset_split
    dataset_output_repo = args.dataset_output_repo
    num_vae_processes = args.num_vae_processes
    t5_num_processes = args.num_t5_processes

    captions_folder = args.captions_folder
    if captions_folder == None:
        captions_folder = images_folder

    output_folder = args.output_folder

    dataset = None
    vae_column_name = None
    t5_column_name = None

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0  # Default to rank 0 if not in a distributed environment

    print(f'rank={rank}')
    if rank == 0:
        if dataset_path != None:
            dataset = load_dataset(dataset_path, split=dataset_split)
        if args.skip_t5_features == False:
            if dataset == None:
                extract_t5_embeddings(repository_path, captions_folder, output_folder)
            else:
                dataset = extract_t5_embeddings_from_dataset(repository_path, dataset, dataset_caption_column, t5_num_processes)
        
        if args.skip_vae_features == False:
            if dataset == None:
                extract_vae_features(repository_path, images_folder, output_folder)
            else:
                dataset = extract_vae_features_from_dataset(repository_path, dataset, dataset_url_column, dataset_image_column, num_vae_processes)

        # optionnaly push the dataset to the hub with the embeddings and latents calculated
        if dataset_output_repo != None:
            # we need to read the embeddings and latents from the disk and add the columns to the dataset
            dataset.push_to_hub(dataset_output_repo, private=False)

        if validation_prompts != None:
            extract_t5_validation_embeddings(repository_path, validation_prompts, output_folder)