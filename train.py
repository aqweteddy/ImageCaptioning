
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np

from utils.preprocess_text import Dictionary
from model import Encoder, Decoder
from utils.dataloader import get_loader
from torchvision import transforms

from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CUDA: {torch.cuda.is_available()}')

crop_size = 224
transform = transforms.Compose([ 
transforms.RandomCrop(crop_size),
transforms.RandomHorizontalFlip(), 
transforms.ToTensor(), 
transforms.Normalize((0.485, 0.456, 0.406), 
                        (0.229, 0.224, 0.225))])
dct = Dictionary()
dct.load_dict('data/vocab.txt')
data_loader = get_loader('data/train2014_resize', 'data/annotations/captions_train2014.json', dct, 
                             transform, batch_size=128,
                             shuffle=True, num_workers=4) 



def train(dct_size, num_layers=1, embed_size=256, hidden_size=512, epochs=10, save_step=1000, lr=0.001, model_save='model/'):
    encoder = Encoder(embed_size=256).to(device)
    decoder = Decoder(embed_size=256, hidden_size=512, dct_size=len(dct), num_layers=1).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    for epoch in range(epochs):
        print(f'epoch {epoch+1}:')
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
train(len(dct), num_layers=1, embed_size=256, hidden_size=512, epochs=5, lr=0.001, save_step=1000, model_save='model/')

