import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        
    def init_context(self, context):
        self.context = context
        # context: [batch_size, enc_seq_len, 2 * hidden_size]
        
    def forward(self, dec_hidden):
        
        gamma_encoder = self.linear_encoder(self.context)
        # gamma_encoder: [batch, enc_seq_len, hidden_size]
        
        gamma_decoder = self.linear_decoder(dec_hidden).unsqueeze(1)
        # gamma_decoder: [batch, 1, hidden_size]
        
        weights = self.linear_v(torch.tanh(gamma_encoder+gamma_decoder)).squeeze(2)
        # [batch, enc_seq_len]
        weights = F.softmax(weights, dim=1)   # [batch, enc_seq_len]
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)  # [batch, 2 * hidden_size]

        return c_t, weights