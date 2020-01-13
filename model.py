# TODO: define model Encoder and Decoder
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # feature extraction
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1) 
        return self.bn(self.linear(features))


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, dct_size, num_layers, max_len=20):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(dct_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.4)
        self.linear = nn.Linear(hidden_size, dct_size)
        self.max_len = max_len
    
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.gru(packed)        
        outputs = self.linear(hiddens[0])
        return outputs
 
    def get_sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_len):
            hiddens, states = self.gru(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
