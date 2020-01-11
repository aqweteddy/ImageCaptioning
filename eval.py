# TODO: input an image output a description

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import Encoder, Decoder

from utils.preprocess_text import Dictionary
# from utils.preprocess_img import

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Cuda: {torch.cuda.is_available()}')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image



def load_model(
    encoder_path,
    decoder_path,
    vocab_size,
    embed_size=256,
    hidden_size=512,
    num_layers=2,
):
    # eval mode (batchnorm uses moving mean/variance)
    encoder = Encoder(embed_size).eval()
    decoder = Decoder(embed_size, hidden_size, vocab_size, num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder


def eval1(img_path,
          vocab,
          encoder,
          decoder
          ):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    image = load_image(img_path, transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.get_sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()

    # convert word_id to word
    result = []
    for idx in sampled_ids:
        word = vocab.id2word[idx]
        result.append(word)
        if word == '<end>':
            break
    return result


if __name__ == '__main__':
    vocab = Dictionary()
    vocab.load_dict('data/vocab.txt')

    encoder, decoder = load_model('model/encoder-5-6000.ckpt',
                                  'model/decoder-5-6000.ckpt',
                                  len(vocab),
                                  embed_size=256,
                                  hidden_size=512,
                                  num_layers=2
                                  )
    with open('data/val.txt', 'w') as f:
        for file in tqdm(os.listdir('data/val2014')):
            if '.jpg' in file:
                try:
                    result = eval1(os.path.join('data/val2014', file),
                            vocab,
                            encoder,
                            decoder)
                    f.writelines([file, ',', ' '.join(result), '\n'])
                except RuntimeError:
                    pass
