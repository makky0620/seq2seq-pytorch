# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
from pprint import pprint
import sys

from util import load_data
from model import EncoderRNN, DecoderRNN
from lang import Lang

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(Lang.EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_lang, output_lang, pairs = load_data()
    hidden_size =256

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    criterion = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    training_pairs = [tensors_from_pair(input_lang, output_lang, random.choice(pairs)) for i in range(75000)]
    
    # epoch 75000
    for epoch in range(2):
        for i, pair in enumerate(training_pairs):
            encoder_hidden = encoder.init_hidden().to(device)
            
            input_tensor = pair[0]
            output_tensor = pair[1]
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0
            
            encoder_output = torch.zeros(hidden_size).to(device) 
            for ei in range(input_tensor.size(0)):
                encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            
            decoder_input = torch.tensor([[Lang.SOS_token]]).to(device)
            decoder_hidden = encoder_hidden

            decoder_output = torch.zeros(output_lang.n_words).to(device) 
            for di in range(output_tensor.size(0)):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, output_tensor[di])

            
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if i % 2000 == 1999:
                print("epoch: {}, iter: {} loss: {}".format(epoch+1, i+1, loss/output_tensor.size(0)))


