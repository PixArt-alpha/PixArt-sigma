<p align="center">
  <img src="asset/logo-sigma.png"  height=120>
</p>


### <div align="center">👉 PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation<div> 

<div align="center">
  <a href="https://pixart-alpha.github.io/PixArt-sigma-project/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2403.04692"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Sigma&color=red&logo=arxiv"></a> &ensp;
  <a href="https://discord.gg/rde6eaE5Ta"><img src="https://img.shields.io/static/v1?label=Discuss&message=Discord&color=purple&logo=discord"></a> &ensp;

</div>

---

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for our paper exploring 
Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation. You can find more visualizations on our [project page](https://pixart-alpha.github.io/PixArt-sigma-project/).

> [**PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation**](https://github.com/PixArt-alpha/PixArt-sigma)<br>
> [Junsong Chen*](https://lawrence-cj.github.io/), [Chongjian Ge*](https://chongjiange.github.io/), 
> [Enze Xie*](https://xieenze.github.io/)&#8224;,
> [Yue Wu*](https://yuewuhkust.github.io/),
> [Lewei Yao](https://scholar.google.com/citations?user=hqDyTg8AAAAJ&hl=zh-CN&oi=ao),
> [Xiaozhe Ren](https://scholar.google.com/citations?user=3t2j87YAAAAJ&hl=en), [Zhongdao Wang](https://zhongdao.github.io/), 
> [Ping Luo](http://luoping.me/), 
> [Huchuan Lu](https://scholar.google.com/citations?hl=en&user=D3nE0agAAAAJ), 
> [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ)
> <br>Huawei Noah’s Ark Lab, DLUT, HKU, HKUST<br>

---
##  Welcome everyone to contribute🔥🔥!!
Learning from the previous [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha) project, 
we will try to keep this repo as simple as possible so that everyone in the PixArt community can use it.

---
## Breaking News 🔥🔥!!
- (🔥 New) Apr. 9, 2024. 💥 [PixArt-Σ checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth) 1024px is released!
- (🔥 New) Apr. 6, 2024. 💥 [PixArt-Σ checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Sigma/tree/main) 256px & 512px are released!
- (🔥 New) Mar. 29, 2024. 💥 [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/) 
training & inference code & toy data are released!!!

---
# 🆚 Compare with [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha)

| Model    | T5 token length | VAE                                                          | 2K/4K |
|----------|-----------------|--------------------------------------------------------------|-------|
| PixArt-Σ | 300             | [SDXL](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) | ✅     |
| PixArt-α | 120             | [SD1.5](https://huggingface.co/stabilityai/sd-vae-ft-ema)    | ❌     |

| Model    | Sample-1                                                                                                                                                      | Sample-2                                                                                                                                                      | Sample-3                                                                                                                                                      |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PixArt-Σ | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample1%CE%A3.webp" width=256> | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample2%CE%A3.webp" width=512> | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample3%CE%A3.webp" width=512> |
| PixArt-α | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample1%CE%B1.webp" width=256> | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample2%CE%B1.webp" width=512> | <img src="https://raw.githubusercontent.com/PixArt-alpha/PixArt-sigma-project/master/static/images/samples/compare_simga_alpha/sample3%CE%B1.webp" width=512> |
| Prompt   | Close-up, gray-haired, bearded man in 60s, observing passersby, in wool coat and **brown beret**, glasses, cinematic.                                         | Body shot, a French woman, Photography, French Streets background, backlight, rim light, Fujifilm.                                                            | Photorealistic closeup video of two pirate ships battling each other as they sail inside **a cup of coffee**.                                                 |

<details><summary>Prompt Details</summary>Sample-1 full prompt: An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a **brown beret** and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film.</details>

# 🔧 Dependencies and Installation

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 2.0.0+cu11.7](https://pytorch.org/)

```bash
conda create -n pixart python==3.9.0
conda activate pixart
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/PixArt-alpha/PixArt-sigma.git
cd PixArt-sigma
pip install -r requirements.txt
```

# 🔥 How to Train
## 1. PixArt Training

**First of all.**

We start a new repo to build a more user friendly and more compatible codebase. The main model structure is the same as PixArt-α, 
you can still develop your function base on the [original repo](https://github.com/PixArt-alpha/PixArt-alpha). 
lso, **This repo will support PixArt-alpha in the future**.

**Now you can train your model without prior feature extraction**.
We reform the data structure in PixArt-α code base, so that everyone can start to **train & inference & visualize**
at the very beginning without any pain. 


### 1.1 Downloading the toy dataset

Download the [toy dataset](https://huggingface.co/datasets/PixArt-alpha/pixart-sigma-toy-dataset) first.
The dataset structure for training is:

```
cd ./pixart-sigma-toy-dataset

Dataset Structure
├──InternImgs/  (images are saved here)
│  ├──000000000000.png
│  ├──000000000001.png
│  ├──......
├──InternData/
│  ├──data_info.json    (meta data)
Optional(👇)
│  ├──img_sdxl_vae_features_1024resolution_ms_new    (run tools/extract_caption_feature.py to generate caption T5 features, same name as images except .npz extension)
│  │  ├──000000000000.npy
│  │  ├──000000000001.npy
│  │  ├──......
│  ├──caption_features_new
│  │  ├──000000000000.npz
│  │  ├──000000000001.npz
│  │  ├──......
│  ├──sharegpt4v_caption_features_new    (run tools/extract_caption_feature.py to generate caption T5 features, same name as images except .npz extension)
│  │  ├──000000000000.npz
│  │  ├──000000000001.npz
│  │  ├──......
```

### 1.2 Download pretrained checkpoint
```bash
# SDXL-VAE, T5 checkpoints
git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pixart_sigma_sdxlvae_T5_diffusers

# PixArt-Sigma checkpoints
python tools/download.py
```

### 1.3 You are ready to train!
Selecting your desired config file from [config files dir](configs/pixart_sigma_config).

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth
          --work-dir output/your_first_pixart-exp \
          --debug
```

# 💻 How to Test
## 1. Quick start with [Gradio](https://www.gradio.app/guides/quickstart)

To get started, first install the required dependencies. Make sure you've downloaded the checkpoint files 
from [models(coming soon)](https://huggingface.co/PixArt-alpha/PixArt-Sigma) to the `output/pretrained_models` folder, 
and then run on your local machine:

```bash
# SDXL-VAE, T5 checkpoints
git lfs install
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pixart_sigma_sdxlvae_T5_diffusers

# PixArt-Sigma checkpoints
python tools/download.py

# demo launch
python scripts/interface.py --model_path output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth --image_size 512 --port 11223
```

## 2. Integration in diffusers
 (Coming soon)


# ⏬ Available Models
All models will be automatically downloaded [here](#12-download-pretrained-checkpoint). You can also choose to download manually from this [url](https://huggingface.co/PixArt-alpha/PixArt-Sigma).

| Model         | #Params | pth                                                                                                                       | Download in OpenXLab |
|:--------------|:--------|:--------------------------------------------------------------------------------------------------------------------------|:---------------------|
| T5 & SDXL-VAE | 4.5B    | [pixart_sigma_sdxlvae_T5_diffusers](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers)                | [coming soon]( )     |
| PixArt-Σ-256  | 0.6B    | [PixArt-Sigma-XL-2-256x256.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-256x256.pth) | [coming soon]( )     |
| PixArt-Σ-512  | 0.6B    | [PixArt-Sigma-XL-2-512-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-512-MS.pth)   | [coming soon]( )     |
| PixArt-Σ-1024 | 0.6B    | [PixArt-Sigma-XL-2-1024-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth) | [coming soon]( )     |


## 💪To-Do List
We will try our best to release

- [x] Training code
- [x] Inference code
- [ ] Model zoo 
- [ ] Diffusers
- [ ] training & inference code of One Step Sampling with [DMD](https://arxiv.org/abs/2311.18828) 

 before 10th, April.