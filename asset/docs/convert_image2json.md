## Tools

# tools/convert_images_to_json.py

This script is used to convert a folder that contains images and caption files to a dataset folder structure required by the PixArt-Sigma training scripts.

Before :
- root
    - image1.png
    - image1.txt
    - image2.jpg
    - image2.txt
    - image3.webp
    - image3.txt

After : 
- out
    - InternData
        - data_info.json
    - InternImgs
        - image1.png
        - image2.jpg
        - image3.webp

The script detects all images with a paired caption and copies these in the InternImgs folder and its prompt in the data_info.json.

The usage is the following:

 `python tools/convert_images_to_json.py [params] images_path output_path`

The caption file extension is by default .txt but the user can change it with the argument `--caption_extension .caption` for example.