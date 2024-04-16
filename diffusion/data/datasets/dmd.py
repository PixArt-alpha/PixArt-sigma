from torch.utils.data import Dataset
import os
import numpy as np
import glob, torch
from diffusion.data.builder import get_data_root_and_path
from PIL import Image
import json
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader


def read_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt_list = f.readlines()
    return prompt_list


# Dataloader for pixart-dmd model
class DMD(Dataset):
    ## rewrite the dataloader to avoid data loading bugs
    def __init__(self,
                 root,
                 transform=None,
                 image_list_json='data_info.json',
                 resolution=512,
                 load_vae_feat=False,
                 load_t5_feat=False,
                 max_samples=None,
                 max_length=120,
                 ):
        '''
        :param root: the root of saving txt features ./data/data/
        :param latent_root: the root of saving latent image pairs
        :param image_list_json:
        :param resolution:
        :param max_samples:
        :param offset:
        :param kwargs:
        '''
        super().__init__()
        DATA_ROOT, root = get_data_root_and_path(root)
        self.root = root
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.load_t5_feat = load_t5_feat
        self.DATA_ROOT = DATA_ROOT
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.max_samples = max_samples
        self.max_lenth = max_length
        self.meta_data_clean = []
        self.img_samples = []
        self.txt_feat_samples = []
        self.noise_samples = []
        self.vae_feat_samples = []
        self.gen_image_samples = []
        self.txt_samples = []

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            meta_data = self.load_json(os.path.join(self.root, json_file))
            self.ori_imgs_nums += len(meta_data)
            meta_data_clean = [item for item in meta_data if item['ratio'] <= 4.5]
            self.meta_data_clean.extend(meta_data_clean)
            self.img_samples.extend([
                os.path.join(self.root.replace('InternData', 'InternImgs'), item['path']) for item in meta_data_clean
            ])
            self.gen_image_samples.extend([
                os.path.join(self.root, 'InternImgs_DMD_images', item['path']) for item in meta_data_clean
            ])
            self.txt_samples.extend([item['prompt'] for item in meta_data_clean])
            self.txt_feat_samples.extend([
                os.path.join(
                    self.root,
                    'caption_features_new',
                    item['path'].rsplit('/', 1)[-1].replace('.png', '.npz')
                ) for item in meta_data_clean
            ])
            self.noise_samples.extend([
                os.path.join(
                    self.root,
                    'InternImgs_DMD_noises',
                    item['path'].rsplit('/', 1)[-1].replace('.png', '.npy')
                ) for item in meta_data_clean
            ])
            self.vae_feat_samples.extend(
                [
                    os.path.join(
                        self.root,
                        'InternImgs_DMD_latents',
                        item['path'].rsplit('/', 1)[-1].replace('.png', '.npy')
                    ) for item in meta_data_clean
                ])

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.latent_feat_loader
        else:
            self.loader = default_loader

    def __len__(self):
        return min(self.max_samples, len(self.img_samples))

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    @staticmethod
    def latent_feat_loader(path):
        return torch.from_numpy(np.load(path))

    def load_ori_img(self, img_path):
        # 加载图像并转换为Tensor
        transform = T.Compose([
            T.Resize(512),  # Image.BICUBIC
            T.CenterCrop(512),
            T.ToTensor(),
        ])
        img = transform(Image.open(img_path))
        img = img * 2.0 - 1.0
        return img

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def getdata(self, index):
        img_gt_path = self.img_samples[index]
        gen_img_path = self.gen_image_samples[index]
        npz_path = self.txt_feat_samples[index]
        txt = self.txt_samples[index]
        npy_path = self.vae_feat_samples[index]
        data_info = {
            'img_hw': torch.tensor([torch.tensor(self.resolution), torch.tensor(self.resolution)], dtype=torch.float32),
            'aspect_ratio': torch.tensor(1.)
        }

        if self.load_vae_feat:
            gen_img = self.loader(npy_path)
        else:
            gen_img = self.loader(gen_img_path)

        attention_mask = torch.ones(1, 1, self.max_lenth)     # 1x1xT
        if self.load_t5_feat:
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info['caption_feature'])     # 1xTx4096
            if 'attention_mask' in txt_info.keys():
                attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]
            if txt_fea.shape[1] < self.max_lenth:
                txt_fea = torch.cat([txt_fea, txt_fea[:, -1:].repeat(1, self.max_lenth-txt_fea.shape[1], 1)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros(1, 1, self.max_lenth-attention_mask.shape[-1])], dim=-1)
            elif txt_fea.shape[1] > self.max_lenth:
                txt_fea = txt_fea[:, :self.max_lenth]
                attention_mask = attention_mask[:, :, :self.max_lenth]
        else:
            txt_fea = txt

        noise = torch.from_numpy(np.load(self.noise_samples[index]))
        img_gt = self.load_ori_img(img_gt_path)

        return {'noise': noise,
                'base_latent': gen_img,
                'latent_path': self.vae_feat_samples[index],
                'text': txt,
                'data_info': data_info,
                'txt_fea': txt_fea,
                'attention_mask': attention_mask,
                'img_gt': img_gt}

    def __getitem__(self, idx):
        data = self.getdata(idx)
        return data
        # for _ in range(20):
        #     try:
        #         data = self.getdata(idx)
        #         return data
        #     except Exception as e:
        #         print(f"Error details: {str(e)}")
        #         idx = np.random.randint(len(self))
        # raise RuntimeError('Too many bad data.')
