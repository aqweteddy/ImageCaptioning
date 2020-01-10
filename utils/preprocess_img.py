import os
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor


def proc1(input_dir: str, image: str, output_dir: str, size):
    with open(os.path.join(input_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img =  img.resize(size, Image.ANTIALIAS)
                img.save(os.path.join(output_dir, image), img.format)

def resize_images(input_dir: str, output_dir: str, size: int, max_workers=4):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(input_dir)
    num_images = len(images)
    proc = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        for image in images:
            proc.append(exe.submit(proc1, input_dir, image, output_dir, size))
        
        for p in tqdm(proc):
            p.result()


