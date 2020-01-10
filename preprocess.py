from utils.preprocess_text import Dictionary
from utils.preprocess_img import resize_images
import argparse
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def main(args):
    if args.img_input_dir:
        print('resizing images....')
        print(f'output to {args.img_output_dir}')
        image_size = [args.image_size, args.image_size]
        resize_images(args.img_input_dir, args.img_output_dir, image_size)
    
    print('building dictionary....')
    if args.caption_path:
        dct = Dictionary()
        coco = COCO(args.caption_path)
        ids = coco.anns.keys()
        
        proc = []
        with ProcessPoolExecutor(max_workers=5) as exe:
            print('tokenizing...')
            for id in tqdm(ids):
                caption = str(coco.anns[id]['caption'])
                dct.add_sentence(caption)
        dct.save_dict(args.dict_path)
        print(f"Total vocabulary size: {len(dct)}")
        print(f"Saved the dictionary to '{args.dict_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, default=None,
                        help='path for train annotation file')
    parser.add_argument('--dict_path', type=str, default=None,
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--img_input_dir', type=str, default=None,
                        help='directory for train images')
    parser.add_argument('--img_output_dir', type=str, default=None,
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    
    args = parser.parse_args()
    main(args)
