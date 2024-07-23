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
from torch.multiprocessing import set_start_method

import sys
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path))

#from train_pixart_diffusers_map import add_embeddings_columns

# vae 16 channels test
# pipe = PixArtSigmaPipeline.from_pretrained('frutiemax/TwistedReality-pixart-512ms', transformer=None, text_encoder=None, torch_dtype=torch.float32)
# #vae = AutoencoderKL.from_pretrained('ostris/vae-kl-f8-d16').to(device='cuda', dtype=torch.float32)
# vae = pipe.vae.to('cuda')
# image_processor = pipe.image_processor

# bw_image = Image.open('6757458677_1d1feb56c9_b.jpg')
# image = Image.new('RGB', bw_image.size)
# image.paste(bw_image)

# image = image_processor.pil_to_numpy(image)
# image = torch.tensor(image, device='cuda', dtype=torch.float32)
# image = torch.moveaxis(image, -1, 1)

# with torch.no_grad():
#     image = image_processor.preprocess(image)
#     latent = vae.encode(image).latent_dist.sample()
#     #latent = latent * vae.config.scaling_factor

#     #latent = latent / vae.config.scaling_factor
#     image = vae.decode(latent).sample
#     image = image_processor.postprocess(image.cpu().detach())
#     image[0].show()

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
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    with lock:
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=None, torch_dtype=torch.float16)
        pipe = pipe.to(device=device)

    def add_t5_columns(batch, pipe, dataset_caption_column):
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
    
    shard = shard.map(add_t5_columns, batched=True, fn_kwargs={'pipe' : pipe, 'dataset_caption_column' : dataset_caption_column})
    del pipe
    flush()
    queue.put(shard)

def extract_t5_embeddings_from_dataset(repository_path, dataset : datasets.Dataset, dataset_caption_column, output_folder):
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

    for p in processes:
        p.join()

    dataset = datasets.concatenate_datasets(results)
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
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

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
        local_rank = int(os.getenv('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)

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
    shard = shard.map(add_vae_column, batched=True)
    queue.put(shard)

def extract_vae_features_from_dataset(repository_path, dataset, dataset_url_column, dataset_images_column, output_folder):
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
        results.append(queue.get())

    for p in processes:
        p.join()

    dataset = datasets.concatenate_datasets(results)
    return dataset    
    
    
class EmbeddingsFeaturesDataset(Dataset):
    def __init__(self, folder, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str):
        self.embeddings_column = embeddings_column
        self.vae_features_column = vae_features_column
        self.files = []
        self.hf_dataset = None
        if hf_dataset == None:
            self.folder = Path(folder)
            self.embeddings_folder = self.folder.joinpath('embeddings')
            self.features_folder = self.folder.joinpath('features')

            for (dirpath, dirnames, filenames) in walk(self.embeddings_folder):
                for filename in filenames:
                    embedding_filepath = self.embeddings_folder.joinpath(filename)
                    features_filepath = self.features_folder.joinpath(Path(filename).stem + '.lat')

                    if os.path.exists(features_filepath):
                        self.files.append((embedding_filepath, features_filepath))
        else:
            self.hf_dataset = hf_dataset
            self.hf_dataset = self.hf_dataset.with_format('torch')
    
    def __len__(self):
        if self.hf_dataset == None:
            return len(self.files)
        else:
            return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        if self.hf_dataset == None:
            embeddings, features = self.files[idx]
            embeddings = torch.load(embeddings)
            features = torch.load(features)
            return embeddings, features
        else:
            item = self.hf_dataset[idx]
            embeddings = tuple([item[self.embeddings_column[index]] for index in range(4)])
            features = item[self.vae_features_column]
            return embeddings, features

class BucketSampler(BatchSampler):
    def __init__(self, files, max_batch_size, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str):
        # initialize the same seed for every process
        random.seed(0)

        self.hf_dataset = hf_dataset
        self.embeddings_column = embeddings_column
        self.vae_features_column = vae_features_column
        self.max_batch_size = max_batch_size
        self.batch_size = max_batch_size
        self.buckets = {}
        self.num_images = 0

        if self.hf_dataset == None:
            self.files = files

            # we need to sort the files so they belong to the same features size
            for i in tqdm.tqdm(range(len(files)), desc='Calculating buckets'):
                file = files[i]
                embeddings, features = file
                latent = torch.load(Path(features))

                width = latent.shape[-1]
                height = latent.shape[-2]

                if not (height, width) in self.buckets.keys():
                    self.buckets[(height, width)] = []
                self.buckets[(height, width)].append(i)
                self.num_images = self.num_images + 1
        
        else:
            self.hf_dataset = self.hf_dataset.with_format('torch')
            for i in tqdm.tqdm(range(len(self.hf_dataset)), desc='Calculating buckets'):
                item = self.hf_dataset[i]
                ratio = float(item['ratio'])

                if not ratio in self.buckets.keys():
                    self.buckets[ratio] = []
                self.buckets[ratio].append(i)
                self.num_images = self.num_images + 1

    def __iter__(self):
        batch = []
        for key, value in self.buckets.items():
            # randomize the value order
            value_copy = copy(value)
            random.shuffle(value_copy)

            for idx in value_copy:
                batch.append(idx)

                if len(batch) == self.max_batch_size:
                    yield batch
                    batch = []
            if batch != []:
                # create a batch with randomly sampled indices
                for idx in batch:
                    value_copy.remove(idx)

                while len(batch) != self.max_batch_size and value_copy != []:
                    idx = random.choice(value_copy)
                    value_copy.remove(idx)
                    batch.append(idx)
                yield batch
                batch = []
    def __len__(self):
        return int(self.num_images / self.max_batch_size)
        
def flush():
    gc.collect()
    torch.cuda.empty_cache()

def validation_and_save(repository_path : str, transformer : PixArtTransformer2DModel, output_folder : str, logger : SummaryWriter,
                        global_step : int, logging_path, validation_prompts : list[str], push_to_hub = False):
    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16,
                                                transformer=transformer).to(device=transformer.device)
    dataset = FilesDataset(Path(output_folder).joinpath('validation_embeddings'), extensions=['.emb'])
    dataloader = DataLoader(dataset)

    index = 0
    for batch, embeddings in tqdm.tqdm(enumerate(dataloader)):
        prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = torch.load(embeddings[0])
        prompt_embeds = prompt_embeds.to(device=transformer.device)
        prompt_attention_mask = prompt_attention_mask.to(device=transformer.device)
        negative_embeds = negative_embeds.to(device=transformer.device)
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device=transformer.device)
        
        with torch.no_grad():
            latents = pipe(negative_prompt=None, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                           prompt_attention_mask=prompt_attention_mask, negative_prompt_attention_mask=negative_prompt_attention_mask, 
                           output_type='latent').images
            image = pipe.vae.decode(latents.to(torch.float16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type='pt')[0]
            logger.add_image(validation_prompts[index], image, global_step)
        del image
        del latents
        flush()
        index = index + 1

    del pipe

    # optionnaly push the pipeline to the hub
    if push_to_hub == True:
        pipe = PixArtSigmaPipeline.from_pretrained(repository_path, transformer=transformer, torch_dtype=torch.float16)
        pipe.push_to_hub(repository_path)
    
    flush()
    
    # save the transformer
    transformer.save_pretrained(Path(logging_path).joinpath(f'{global_step}'))
    
def train(output_folder : str, num_epochs : int, batch_size : int, repository_path : str, learning_rate : float, steps_per_validation : int, epochs_per_validation : int,
          blank_transformer : bool, hf_dataset : datasets.Dataset, embeddings_column : str, vae_features_column : str, validation_prompts : list[str], push_transformer_to_hub : bool):
    dataset = EmbeddingsFeaturesDataset(output_folder, hf_dataset, embeddings_column, vae_features_column)
    sampler = BucketSampler(dataset.files, batch_size, hf_dataset, embeddings_column, vae_features_column)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    # load the transformer
    transformer = PixArtTransformer2DModel.from_pretrained(repository_path, subfolder='transformer').to(device='cuda')
    if blank_transformer == True:
        transformer = PixArtTransformer2DModel.from_config(transformer.config)
    transformer.gradient_checkpointing = True
    transformer.training = True
    scheduler = DDPMScheduler.from_pretrained(repository_path, subfolder='scheduler')

    # load the arguments optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(transformer.parameters(), lr=learning_rate)

    # prepare the logging dir
    date = datetime.now().timestamp()
    logs_dir = Path(output_folder).joinpath(f'logs{date}')
    logger = SummaryWriter(logs_dir)

    resolution = transformer.config.interpolation_scale * 512
    # prepare multi-gpu training
    accelerator = Accelerator()
    transformer = transformer.to(accelerator.device)
    #dataloader, transformer, optimizer = accelerator.prepare(dataloader, transformer, optimizer)
    dataloader = accelerator.prepare(dataloader)
    transformer = accelerator.prepare(transformer)
    optimizer = accelerator.prepare(optimizer)
    added_cond_kwargs = {"resolution": resolution, "aspect_ratio": None}

    global_step = 0
    dtype = accelerator.unwrap_model(transformer).dtype
    device = accelerator.unwrap_model(transformer).device
    for epoch in tqdm.tqdm(range(num_epochs)):
        if epoch % epochs_per_validation == 0 and accelerator.is_main_process:
            validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir, validation_prompts)
        
        for batch_embeds, batch_features in tqdm.tqdm(dataloader, desc='Batch:'):
            if global_step % steps_per_validation == 0 and accelerator.is_main_process:
                validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir, validation_prompts)

            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = batch_embeds
                
                # strip the second dimension
                prompt_embeds = torch.squeeze(prompt_embeds, dim=1).to(dtype=dtype)
                prompt_attention_mask = torch.squeeze(prompt_attention_mask, dim=1).to(dtype=dtype)
                negative_prompt_embeds = torch.squeeze(negative_prompt_embeds, dim=1).to(dtype=dtype)
                negative_prompt_attention_mask = torch.squeeze(negative_prompt_attention_mask, dim=1).to(dtype=dtype)
                batch_features = torch.squeeze(batch_features, dim=1).to(dtype=dtype)
                batch_size = batch_features.shape[0]
    
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
                timesteps = timestep.expand(batch_size)
    
                noise = randn_tensor(batch_features.shape, device=device, dtype=dtype)
                noisy_latents = scheduler.add_noise(batch_features, noise, timesteps)
                latent_model_input = noisy_latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, timesteps)

            with accelerator.accumulate(transformer):
                noise_pred = accelerator.unwrap_model(transformer)(latent_model_input,
                                        encoder_hidden_states=prompt_embeds,
                                        encoder_attention_mask=prompt_attention_mask,
                                        timestep=timesteps,
                                        added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]
                loss = loss_fn(noise_pred, noise).to(dtype=dtype)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                transformer.zero_grad()

            # some logging statistics
            logger.add_scalar('loss', loss.detach().item(), global_step)
            global_step = global_step + 1
    
    # final save
    if accelerator.is_main_process:
        validation_and_save(repository_path, accelerator.unwrap_model(transformer), output_folder, logger, global_step, logs_dir, validation_prompts, push_transformer_to_hub)
            

if __name__ == '__main__':
    # if on windows, try to set to gloo for distributed training
    try:
        if dist.is_available() and not dist.is_initialized():
            if os.name == 'nt':
                torch.distributed.init_process_group(backend='gloo')
            else:
                print('torch.distributed.init_process_group(backend=''nccl'')')
                torch.distributed.init_process_group(backend='nccl')
    except:
        pass
    set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--repository_path', required=True, type=str)
    parser.add_argument('--images_folder', required=False, type=str, default=None)
    parser.add_argument('--captions_folder', required=False, type=str, default=None)
    parser.add_argument('-l', '--validation_prompts', nargs='+', required=False, default=None)
    parser.add_argument('--output_folder', required=True, type=str)
    parser.add_argument('--batch_size', required=False, type=int, default=8)
    parser.add_argument('--num_epochs', required=False, type=int, default=5)
    parser.add_argument('--learning_rate', required=False, type=float, default=1e-5)
    parser.add_argument('--steps_per_validation', required=False, type=int, default=500)
    parser.add_argument('--epochs_per_validation', required=False, type=int, default=3)
    parser.add_argument('--skip_t5_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--skip_vae_features', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--blank_transformer', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--dataset_image_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_path', required=False, type=str, default=None)
    parser.add_argument('--dataset_url_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_caption_column', required=False, type=str, default=None)
    parser.add_argument('--dataset_split', required=False, type=str, default='train')
    parser.add_argument('--dataset_output_repo', required=False, type=str, default=None)
    parser.add_argument('--push_transformer_to_hub', action=argparse.BooleanOptionalAction, default=False)

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
    push_transformer_to_hub = args.push_transformer_to_hub

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

    
    if dataset_path != None:
        dataset = load_dataset(dataset_path, split='train')
    if args.skip_t5_features == False:
        if dataset == None:
            extract_t5_embeddings(repository_path, captions_folder, output_folder)
        elif rank == 0:
            dataset = extract_t5_embeddings_from_dataset(repository_path, dataset, dataset_caption_column, output_folder)
    
    if args.skip_vae_features == False:
        if dataset == None:
            extract_vae_features(repository_path, images_folder, output_folder)
        elif rank == 0:
            dataset = extract_vae_features_from_dataset(repository_path, dataset, dataset_url_column, dataset_image_column, output_folder)

    # optionnaly push the dataset to the hub with the embeddings and latents calculated
    if dataset_output_repo != None:
        # we need to read the embeddings and latents from the disk and add the columns to the dataset
        dataset.push_to_hub(dataset_output_repo, private=False)

    if validation_prompts != None and rank == 0:
        extract_t5_validation_embeddings(repository_path, validation_prompts, output_folder)
    
    # save the dataset to the disk so other processes can use it
    if args.skip_t5_features == False and args.skip_vae_features == False and dataset != None and rank == 0 and dataset_output_repo == None:
        dataset_path = Path(output_folder).joinpath('dataset.arrow')
        dataset.save_to_disk(dataset_path)

    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except:
        pass

    if dataset_output_repo != None:
        dataset = load_dataset(dataset_output_repo, split='train')
    elif args.skip_t5_features == False or args.skip_vae_features == False:
        dataset_path = Path(output_folder).joinpath('dataset.arrow')
        if os.path.exists(dataset_path):
            dataset = datasets.Dataset.load_from_disk(dataset_path)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    steps_per_validation = args.steps_per_validation
    epochs_per_validation = args.epochs_per_validation
    blank_transformer = args.blank_transformer

    pipe = PixArtSigmaPipeline.from_pretrained(repository_path, text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
    interpolation_scale = pipe.transformer.config.interpolation_scale
    checkpoint_resolution = 512 * interpolation_scale
    del pipe
    flush()

    train(output_folder, num_epochs, batch_size, repository_path, learning_rate, steps_per_validation, epochs_per_validation, blank_transformer, dataset,
          ['t5_prompt_embeds', 't5_prompt_attention_mask', 't5_negative_embeds', 't5_negative_prompt_attention_mask'],
          f'vae_{checkpoint_resolution}px', validation_prompts, push_transformer_to_hub)