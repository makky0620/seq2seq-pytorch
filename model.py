# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# class Seq2seq(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Seq2seq, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.encoder = EncoderRNN(input_size, hidden_size)
#         self.decoder = DecoderRNN(hidden_size, output_size)

#     def forward(self, input, hidden):
#         for word in input:
#             output, hidden = self.encoder(word, hidden)
#             print(output[0, 0])
        
#         for word in
        
#         # output, hidden = self.decoder(output, hidden)
#         # return output, hidden
#         return 0, 0
    
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size)


