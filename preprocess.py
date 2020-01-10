from utils.preprocess_text import Dictionary
from utils.preprocess_img import resize_images
import argparse
from pycocotools.coco import COCO


def main(args):
    if args.image_dir:
        image_size = [args.image_size, args.image_size]
        resize_images(args.image_dir, args.output_dir, image_size)
    
    if args.caption_path:
        dct = Dictionary()
        coco = COCO(args.caption_path)
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            dct.add_sentence(caption)
            if (i+1) % 1000:
                print(f'Tokenizing {i+1}')
        dct.save(args.dict_path)
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
