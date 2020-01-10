import os
from PIL import Image


def resize_images(input_dir: str, output_dir: str, size: int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img =  img.resize(size, Image.ANTIALIAS)
                img.save(os.path.join(output_dir, image), img.format)
        if (i+1) % 100 == 0:
            print(f'resize image: No.{i+1}')
