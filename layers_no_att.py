import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import layers

class bilstm_Output(nn.Module):

    def __init__(self, hidden_size, drop_prob):
        super(bilstm_Output, self).__init__()
        # self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = layers.RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        # self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self , mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 =  self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))

        logits_2 = self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    
class lstm_Output(nn.Module):

    def __init__(self, hidden_size, drop_prob):
        super(lstm_Output, self).__init__()
        # self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(hidden_size, 1)

        self.rnn = layers.RNNEncoder(input_size=2*hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        # self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(hidden_size, 1)

    def forward(self , mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 =  self.mod_linear_1(mod)

        mod_2 = self.rnn(mod, mask.sum(-1))

        logits_2 = self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    