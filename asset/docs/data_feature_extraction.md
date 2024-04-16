# 📕 Data Preparation

### 1.Downloading the toy dataset

Download the [toy dataset](https://huggingface.co/datasets/PixArt-alpha/pixart-sigma-toy-dataset) first.
The dataset structure for training is:

```
cd your_project_path/pixart-sigma-toy-dataset

Dataset Structure
├──InternImgs/  (images are saved here)
│  ├──000000000000.png
│  ├──000000000001.png
│  ├──......
├──InternData/
│  ├──data_info.json    (meta data)
Optional(👇)
│  ├──img_sdxl_vae_features_512resolution_ms_new    (run tools/extract_caption_feature.py to generate caption T5 features, same name as images except .npz extension)
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
### You are already able to run the [training code](https://github.com/PixArt-alpha/PixArt-sigma#12-download-pretrained-checkpoint)

---
## Optional(👇)
> [!IMPORTANT]  
> You don't have to extract following feature to do the training, BUT
> 
> if you want to train with **faster speed** and **lower GPU occupancy**, you can pre-process all the VAE & T5 features

### 2. Extract VAE features

```bash
python tools/extract_features.py --run_vae_feature_extract \
                                 --multi_scale \
                                 --img_size=512 \
                                 --dataset_root=pixart-sigma-toy-dataset/InternData \
                                 --vae_json_file=data_info.json \
                                 --vae_models_dir=madebyollin/sdxl-vae-fp16-fix \
                                 --vae_save_root=pixart-sigma-toy-dataset/InternData
```
**SDXL-VAE** features will be saved at: `pixart-sigma-toy-dataset/InternData/img_sdxl_vae_features_512resolution_ms_new` 
 as shown in the [DataTree](#1downloading-the-toy-dataset).
They will be later used in [InternalData_ms.py](https://github.com/PixArt-alpha/PixArt-sigma/blob/d5adc756dd6a8b64f1f0aaa1d266e90949e873c0/diffusion/data/datasets/InternalData_ms.py#L242)

### 3. Extract T5 features (prompt)

```bash
python tools/extract_features.py --run_t5_feature_extract \
                                 --max_length=300 \
                                 --t5_json_path=pixart-sigma-toy-dataset/InternData/data_info.json \
                                 --t5_models_dir=PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers \
                                 --caption_label=prompt \
                                 --t5_save_root=pixart-sigma-toy-dataset/InternData
```
**T5 features** will be saved at: `pixart-sigma-toy-dataset/InternData/caption_features_new`
as shown in the [DataTree](#1downloading-the-toy-dataset).
They will be later used in [InternalData_ms.py](https://github.com/PixArt-alpha/PixArt-sigma/blob/d5adc756dd6a8b64f1f0aaa1d266e90949e873c0/diffusion/data/datasets/InternalData_ms.py#L227)

---
> [!TIP]  
> Ignore it if you don't have `sharegpt4v` in your data_info.json

### 3.1. Extract T5 features (sharegpt4v)

```bash
python tools/extract_features.py --run_t5_feature_extract \
                                 --max_length=300 \
                                 --t5_json_path=pixart-sigma-toy-dataset/InternData/data_info.json \
                                 --t5_models_dir=PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers \
                                 --caption_label=sharegpt4v \
                                 --t5_save_root=pixart-sigma-toy-dataset/InternData
```
**T5 features** will be saved at: `pixart-sigma-toy-dataset/InternData/caption_features_new`
as shown in the [DataTree](#1downloading-the-toy-dataset).
They will be later used in [InternalData_ms.py](https://github.com/PixArt-alpha/PixArt-sigma/blob/d5adc756dd6a8b64f1f0aaa1d266e90949e873c0/diffusion/data/datasets/InternalData_ms.py#L234)

