
import os
import parser

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

from utils.preprocess_text import Dictionary
from model import Encoder, Decoder
from utils.dataloader import get_loader
from torchvision import transforms

from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
parser.add_argument('--vocab_path', type=str, default='data/vocab.txt', help='path for vocabulary wrapper')
parser.add_argument('--image_dir', type=str, default='data/train2014_resize', help='directory for resized images')
parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
parser.add_argument('--save_ckpt', type=int , default=1000, help='step size for saving trained models(check points)')

# Model parameters
parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CUDA: {torch.cuda.is_available()}')

crop_size = args.crop_size
transform = transforms.Compose([ 
transforms.RandomCrop(crop_size),
transforms.RandomHorizontalFlip(), 
transforms.ToTensor(), 
transforms.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225))])
dct = Dictionary()
dct.load_dict(args.vocab_path)
print(f'Dict Size {len(dct)}')
data_loader = get_loader(args.image_dir, args.caption_path, dct, 
                             transform, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 



def train(dct_size, embed_size=256, hidden_size=512, epochs=10, num_layers=1, save_step=1000, lr=0.001, model_save='model/'):
    encoder = Encoder(embed_size=embed_size).to(device)
    decoder = Decoder(embed_size=embed_size, hidden_size=hidden_size, dct_size=len(dct), num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        print(f'epoch {epoch+1}/{epochs}: ')
        for i, (images, captions, lengths) in enumerate(tqdm(data_loader)):
        # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_save, 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_save, 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


train(len(dct), num_layers=args.num_layers, 
                embed_size=args.embed_size, 
                hidden_size=args.hidden_size, 
                epochs=args.num_epochs, 
                lr=args.learning_rate, 
                save_step=args.save_ckpt, 
                model_save=args.model_path)

