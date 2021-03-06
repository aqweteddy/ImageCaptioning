# TODO: define dataloader
import os

import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import dataloader


class CocoDataset(data.Dataset):
    def __init__(self, img_path: str, json: str, dct: object, transform):
        """
        root: image directory.
        json: coco annotation file path.
        vocab: vocabulary wrapper.
        transform: image transformer.
        """
        self.root = img_path
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.dct = dct
        self.transform = transform
    
    def __getitem__(self, key):
        """
        return pair(img, caption)
        """
        
        ann_id = self.ids[key]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        tokens = nltk.tokenize.word_tokenize(str(caption))
        caption = [int(self.dct['<start>'])]
        caption.extend([int(self.dct[token]) for token in tokens])
        caption.append(int(self.dct['<end>']))
        target = torch.Tensor(caption)
        return img, target
    
    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, dct, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(img_path=root,
                       json=json,
                       dct=dct,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader 
