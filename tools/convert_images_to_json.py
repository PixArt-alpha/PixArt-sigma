import sys
from pathlib import Path
from os import walk
from os import path
from PIL import Image
import json
import tqdm

def print_usage():
    print('convert_images_to_json [params] images_path output_path')
    print('--caption_extension')

def main():
    args = sys.argv
    if len(args) < 3:
        print_usage()
        return
    
    input_folder = args[1]
    output_folder = args[-1]

    caption_extension = '.txt'
    try:
        caption_arg = args.index('--caption_extension')
        caption_extension = args[caption_arg + 1]
    except:
        pass

    # create a folder with the output path
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # create a InternData and a InternImgs inside the output path
    intern_data_folder = output_folder.joinpath('InternData')
    intern_data_folder.mkdir(parents=True, exist_ok=True)

    intern_imgs_folder = output_folder.joinpath('InternImgs')
    intern_imgs_folder.mkdir(parents=True, exist_ok=True)

    # create a data_info.json inside InternData
    data_info_path = intern_data_folder.joinpath('data_info.json')

    # create a table which will contain all the entries
    json_entries = []
    with open(data_info_path, 'w') as json_file:
        for (dirpath, dirnames, filenames) in walk(input_folder):
            for filename in tqdm.tqdm(filenames):
                if not caption_extension in filename:
                    continue

                # check if an image exists for this caption
                image_filename = filename[:-len(caption_extension)]

                for image_extension in ['.jpg', '.png', '.jpeg', 'webp']:
                    image_path = Path(dirpath).joinpath(image_filename + image_extension)
                    if path.exists(image_path):
                        write_entry(json_entries, dirpath, image_path, Path(dirpath).joinpath(filename), image_filename + image_extension, intern_imgs_folder)
                        break
    
        # use the entries
        json_file.write(json.dumps(json_entries))

def write_entry(json_entries, folder, image_path, caption_path, image_filename, intern_imgs_path):
    # open the file containing the prompt
    with open(caption_path) as prompt_file:
        prompt = prompt_file.read()

        # read the images info
        image = Image.open(image_path)
        image_width = image.width
        image_height = image.height
        ratio = image_height / image_width

        entry = {}
        entry['width'] = image_width
        entry['height'] = image_height
        entry['ratio'] = ratio
        entry['path'] = image_filename
        entry['prompt'] = prompt
        entry['sharegpt4v'] = ''

        json_entries.append(entry)

        # make sure to copy the image to the internimgs folder with the new filename!
        image_output_path = intern_imgs_path.joinpath(image_filename)
        image.save(image_output_path)

if __name__ == '__main__':
    main()