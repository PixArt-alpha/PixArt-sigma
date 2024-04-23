import os
import random
from PIL import Image
import json
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from diffusion.data.builder import get_data_path, DATASETS
from diffusion.utils.logger import get_root_logger


# helper function for replacing image extensions with an another
def replace_img_ext(path, dst_ext: str) -> str:
    return path.replace('.png', dst_ext).replace('.jpg', dst_ext).replace('.webp', dst_ext)


@DATASETS.register_module()
class InternalData(Dataset):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 load_mask_index=False,
                 max_length=120,
                 config=None,
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.max_lenth = max_length
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.prompt_samples = []

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, 'partition', json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([os.path.join(self.root.replace('InternData', "InternImgs"), item['path']) for item in meta_data_clean])
            self.txt_feat_samples.extend([
                os.path.join(
                    self.root,
                    'caption_feature_wmask',
                    replace_img_ext('_'.join(item['path'].rsplit('/', 1)), '.npz')
            ) for item in meta_data_clean
            ])
            self.vae_feat_samples.extend([
                os.path.join(
                    self.root,
                    f'img_vae_features_{resolution}resolution/noflip',
                    replace_img_ext('_'.join(item['path'].rsplit('/', 1)), '.npy')
                ) for item in meta_data_clean
            ])
            self.prompt_samples.extend([item['prompt'] for item in meta_data_clean])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f"T5 max token length: {self.max_lenth}")

    def getdata(self, index):
        img_path = self.img_samples[index]
        npz_path = self.txt_feat_samples[index]
        npy_path = self.vae_feat_samples[index]
        prompt = self.prompt_samples[index]
        data_info = {
            'img_hw': torch.tensor([torch.tensor(self.resolution), torch.tensor(self.resolution)], dtype=torch.float32),
            'aspect_ratio': torch.tensor(1.)
        }

        img = self.loader(npy_path) if self.load_vae_feat else self.loader(img_path)
        txt_info = np.load(npz_path)
        txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
        attention_mask = torch.ones(1, 1, txt_fea.shape[1])     # 1x1xT
        if 'attention_mask' in txt_info.keys():
            attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
        if txt_fea.shape[1] != self.max_lenth:
            txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_lenth-txt_fea.shape[1], 1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_lenth-attention_mask.shape[-1])], dim=-1)

        if self.transform:
            img = self.transform(img)

        data_info['prompt'] = prompt
        return img, txt_fea, attention_mask, data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    def load_ori_img(self, img_path):
        # 加载图像并转换为Tensor
        transform = T.Compose([
            T.Resize(256),  # Image.BICUBIC
            T.CenterCrop(256),
            T.ToTensor(),
        ])
        return transform(Image.open(img_path))

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

@DATASETS.register_module()
class InternalDataSigma(Dataset):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 load_t5_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 mask_type='null',
                 load_mask_index=False,
                 real_prompt_ratio=1.0,
                 max_length=300,
                 config=None,
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.load_t5_feat = load_t5_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        self.mask_type = mask_type
        self.real_prompt_ratio = real_prompt_ratio
        self.max_lenth = max_length
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_samples = []
        self.sharegpt4v_txt_samples = []
        self.txt_feat_samples = []
        self.vae_feat_samples = []
        self.mask_index_samples = []
        self.gpt4v_txt_feat_samples = []
        self.weight_dtype = torch.float16 if self.real_prompt_ratio > 0 else torch.float32
        logger = get_root_logger() if config is None else get_root_logger(os.path.join(config.work_dir, 'train_log.log'))
        logger.info(f"T5 max token length: {self.max_lenth}")
        logger.info(f"ratio of real user prompt: {self.real_prompt_ratio}")

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, json_file))
            logger.info(f"{json_file} data volume: {len(meta_data)}")
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                os.path.join(self.root.replace('InternData', 'InternImgs'), item['path']) for item in meta_data_clean
            ])
            self.txt_samples.extend([item['prompt'] for item in meta_data_clean])
            self.sharegpt4v_txt_samples.extend([item['sharegpt4v'] if 'sharegpt4v' in item else '' for item in meta_data_clean])
            self.txt_feat_samples.extend([
                os.path.join(
                    self.root,
                    'caption_features_new',
                    item['path'].rsplit('/', 1)[-1].replace('.png', '.npz')
                ) for item in meta_data_clean
            ])
            self.gpt4v_txt_feat_samples.extend([
                os.path.join(
                    self.root,
                    'sharegpt4v_caption_features_new',
                    item['path'].rsplit('/', 1)[-1].replace('.png', '.npz')
                ) for item in meta_data_clean
            ])
            self.vae_feat_samples.extend(
                [
                os.path.join(
                    self.root,
                    f'img_sdxl_vae_features_{resolution}resolution_new',
                    item['path'].rsplit('/', 1)[-1].replace('.png', '.npy')
                ) for item in meta_data_clean
            ])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

    def getdata(self, index):
        img_path = self.img_samples[index]
        real_prompt = random.random() < self.real_prompt_ratio
        npz_path = self.txt_feat_samples[index] if real_prompt else self.gpt4v_txt_feat_samples[index]
        txt = self.txt_samples[index] if real_prompt else self.sharegpt4v_txt_samples[index]
        npy_path = self.vae_feat_samples[index]
        data_info = {'img_hw': torch.tensor([torch.tensor(self.resolution), torch.tensor(self.resolution)], dtype=torch.float32),
                     'aspect_ratio': torch.tensor(1.)}

        if self.load_vae_feat:
            img = self.loader(npy_path)
        else:
            img = self.loader(img_path)

        attention_mask = torch.ones(1, 1, self.max_lenth)     # 1x1xT
        if self.load_t5_feat:
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
            if 'attention_mask' in txt_info.keys():
                attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            if txt_fea.shape[1] != self.max_lenth:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_lenth-txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_lenth-attention_mask.shape[-1])], dim=-1)
        else:
            txt_fea = txt

        if self.transform:
            img = self.transform(img)

        data_info["mask_type"] = self.mask_type
        return img, txt_fea, attention_mask.to(torch.int16), data_info

    def __getitem__(self, idx):
        for _ in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                print(f"Error details {self.img_samples[idx]}: {str(e)}")
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    def load_ori_img(self, img_path):
        # 加载图像并转换为Tensor
        transform = T.Compose([
            T.Resize(256),  # Image.BICUBIC
            T.CenterCrop(256),
            T.ToTensor(),
        ])
        img = transform(Image.open(img_path))
        return img

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

