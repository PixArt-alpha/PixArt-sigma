<p align="center">
  <img src="asset/logo-sigma.png"  height=120>
</p>

### <div align="center">👉 PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation<div> 

<div align="center">
  <a href="https://pixart-alpha.github.io/PixArt-sigma-project/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2403.04692"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Sigma&color=red&logo=arxiv"></a> &ensp;
  <a href="https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma"><img src="https://img.shields.io/static/v1?label=Demo&message=HuggingFace&color=yellow"></a> &ensp;
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
- (🔥 New) Apr. 24, 2024. 💥 [🧨 diffusers](https://github.com/huggingface/diffusers/pull/7654) support us now! Congrats!🎉. Remember to update your [diffusers checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-2K-MS/tree/main/transformer) once to make it available.
- (🔥 New) Apr. 24, 2024. 💥 [LoRA code](asset/docs/pixart_lora.md) is released!!
- (✅ New) Apr. 23, 2024. 💥 [PixArt-Σ 2K ckpt](#12-download-pretrained-checkpoint) is released!!
- (✅ New) Apr. 16, 2024. 💥 [PixArt-Σ Online Demo](https://huggingface.co/spaces/PixArt-alpha/PixArt-Sigma) is available!!
- (✅ New) Apr. 16, 2024. 💥 PixArt-α-DMD One Step Generator [training code](asset/docs/pixart_dmd.md) are all released!
- (✅ New) Apr. 11, 2024. 💥 [PixArt-Σ Demo](#3-pixart-demo) & [PixArt-Σ Pipeline](#2-integration-in-diffusers)! PixArt-Σ supports `🧨 diffusers` using [patches](scripts/diffusers_patches.py) for fast experience!
- (✅ New) Apr. 10, 2024. 💥 PixArt-α-DMD one step sampler [demo code](app/app_pixart_dmd.py) & [PixArt-α-DMD checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512) 512px are released!
- (✅ New) Apr. 9, 2024. 💥 [PixArt-Σ checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth) 1024px is released!
- (✅ New) Apr. 6, 2024. 💥 [PixArt-Σ checkpoint](https://huggingface.co/PixArt-alpha/PixArt-Sigma/tree/main) 256px & 512px are released!
- (✅ New) Mar. 29, 2024. 💥 [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/) training & inference code & toy data are released!!!

---
## Contents
-Main
* [Weak-to-Strong](#-compare-with-pixart-α)
* [Training](#-how-to-train)
* [Inference](#-how-to-test)
* [Use diffusers](#2-integration-in-diffusers)
* [Launch Demo](#3-pixart-demo)
* [Available Models](#-available-models)

-Guidance
* [Feature extraction* (Optional)](asset/docs/data_feature_extraction.md)
* [One step Generation (DMD)](asset/docs/pixart_dmd.md)
* [LoRA & DoRA](asset/docs/pixart_lora.md)
* [LCM: coming soon]
* [ControlNet: coming soon]
* [ComfyUI: coming soon]
* [Data reformat* (Optional)](asset/docs/convert_image2json.md)

-Others
* [Acknowledgement](#acknowledgements)
* [Citation](#bibtex)
* [TODO](#to-do-list)
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
- [PyTorch >= 2.0.1+cu11.7](https://pytorch.org/)

```bash
conda create -n pixart python==3.9.0
conda activate pixart
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

git clone https://github.com/PixArt-alpha/PixArt-sigma.git
cd PixArt-sigma
pip install -r requirements.txt
```

---
# 🔥 How to Train
## 1. PixArt Training

**First of all.**

We start a new repo to build a more user friendly and more compatible codebase. The main model structure is the same as PixArt-α, 
you can still develop your function base on the [original repo](https://github.com/PixArt-alpha/PixArt-alpha). 
lso, **This repo will support PixArt-alpha in the future**.

> [!TIP]  
> **Now you can train your model without prior feature extraction**.
> We reform the data structure in PixArt-α code base, so that everyone can start to **train & inference & visualize**
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
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers

# PixArt-Sigma checkpoints
python tools/download.py # environment eg. HF_ENDPOINT=https://hf-mirror.com can use for HuggingFace mirror
```

### 1.3 You are ready to train!
Selecting your desired config file from [config files dir](configs/pixart_sigma_config).

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=12345 \
          train_scripts/train.py \
          configs/pixart_sigma_config/PixArt_sigma_xl2_img512_internalms.py \
          --load-from output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth \
          --work-dir output/your_first_pixart-exp \
          --debug
```

---
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

> [!IMPORTANT]  
> Upgrade your `diffusers` to make the `PixArtSigmaPipeline` available!
> ```bash
> pip install git+https://github.com/huggingface/diffusers
> ```
> 
> For `diffusers<0.28.0`, check this [script](scripts/diffusers_patches.py) for help.
```python
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

transformer = Transformer2DModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
    subfolder='transformer', 
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
image.save("./catcus.png")
```

## 3. PixArt Demo
```bash
pip install git+https://github.com/huggingface/diffusers

# PixArt-Sigma 1024px
DEMO_PORT=12345 python app/app_pixart_sigma.py

# PixArt-Sigma One step Sampler(DMD)
DEMO_PORT=12345 python app/app_pixart_dmd.py
```
Let's have a look at a simple example using the `http://your-server-ip:12345`.


## 4. Convert .pth checkpoint into diffusers version
Directly download from [Hugging Face](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)

or run with:
```bash
pip install git+https://github.com/huggingface/diffusers

python tools/convert_pixart_to_diffusers.py --orig_ckpt_path output/pretrained_models/PixArt-Sigma-XL-2-1024-MS.pth --dump_path output/pretrained_models/PixArt-Sigma-XL-2-1024-MS --only_transformer=True --image_size=1024 --version sigma
```

---
# ⏬ Available Models
All models will be automatically downloaded [here](#12-download-pretrained-checkpoint). You can also choose to download manually from this [url](https://huggingface.co/PixArt-alpha/PixArt-Sigma).

| Model            | #Params | Checkpoint path                                                                                                                                                                                                                            | Download in OpenXLab |
|:-----------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|
| T5 & SDXL-VAE    | 4.5B    | Diffusers: [pixart_sigma_sdxlvae_T5_diffusers](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers)                                                                                                                      | [coming soon]( )     |
| PixArt-Σ-256     | 0.6B    | pth: [PixArt-Sigma-XL-2-256x256.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-256x256.pth) <br/> Diffusers: [PixArt-Sigma-XL-2-256x256](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-256x256) | [coming soon]( )     |
| PixArt-Σ-512     | 0.6B    | pth: [PixArt-Sigma-XL-2-512-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-512-MS.pth) <br/> Diffusers: [PixArt-Sigma-XL-2-512-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-512-MS)     | [coming soon]( )     |
| PixArt-α-512-DMD | 0.6B    | Diffusers: [PixArt-Alpha-DMD-XL-2-512x512](https://huggingface.co/PixArt-alpha/PixArt-Alpha-DMD-XL-2-512x512)                                                                                                                              | [coming soon]( )     |
| PixArt-Σ-1024    | 0.6B    | pth: [PixArt-Sigma-XL-2-1024-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth) <br/> Diffusers: [PixArt-Sigma-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS) | [coming soon]( )     |
| PixArt-Σ-2K      | 0.6B    | pth: [PixArt-Sigma-XL-2-2K-MS.pth](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-2K-MS.pth) <br/> Diffusers: [PixArt-Sigma-XL-2-2K-MS](https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-2K-MS)         | [coming soon]( )     |

---
## 💪To-Do List
We will try our best to release

- [x] Training code
- [x] Inference code
- [x] Inference code of One Step Sampling with [DMD](https://arxiv.org/abs/2311.18828) 
- [x] Model zoo (256/512/1024/2K)
- [x] Diffusers (for fast experience)
- [x] Training code of One Step Sampling with [DMD](https://arxiv.org/abs/2311.18828) 
- [x] Diffusers (stable official version: https://github.com/huggingface/diffusers/pull/7654)
- [x] LoRA training & inference code
- [ ] Model zoo (KV Compress...)
- [ ] ControlNet training & inference code

---
# 🤗Acknowledgements
- Thanks to [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha), [DiT](https://github.com/facebookresearch/DiT) and [OpenDMD](https://github.com/Zeqiang-Lai/OpenDMD) for their wonderful work and codebase!
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for their wonderful technical support and awesome collaboration!
- Thanks to [Hugging Face](https://github.com/huggingface) for sponsoring the nicely demo!

# 📖BibTeX
    @misc{chen2024pixartsigma,
      title={PixArt-\Sigma: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation},
      author={Junsong Chen and Chongjian Ge and Enze Xie and Yue Wu and Lewei Yao and Xiaozhe Ren and Zhongdao Wang and Ping Luo and Huchuan Lu and Zhenguo Li},
      year={2024},
      eprint={2403.04692},
      archivePrefix={arXiv},
      primaryClass={cs.CV}

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PixArt-alpha/PixArt-Sigma&type=Date)](https://star-history.com/#PixArt-alpha/PixArt-Sigma&Date)
