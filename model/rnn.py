import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from model.attentions import bahdanau_attention

class rnn_encoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_encoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=True)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=True)
            
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size * self.config.dec_num_layers) # initial hidden state in  decoder: S0
        
        if self.config.cell == "lstm":
            self.fc_cell = nn.Linear(self.hidden_size * 2, self.hidden_size * self.config.dec_num_layers) # initial cell in decoder
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, inputs, lengths=None):
        if lengths:
            embs = pack(self.dropout(self.embedding(inputs)), lengths)
            outputs, state = self.rnn(embs)
            outputs, lengths_ = unpack(outputs)
        else:
            embs = self.dropout(self.embedding(inputs))
            outputs, state = self.rnn(embs)
        # outputs: (seq_len, batch_size, 2 * hidden_size)
        # state for GRU: (2 ∗ num_layers, batch_size, hidden_size)
        # state for LSTM: (state,cell)
        # cell: (2 ∗ num_layers, batch_size, hidden_size)
        
        # outputs: [max_src_len, batch_size, hidden_size]
        
        if self.config.cell == 'gru':
            #state [-2, :, : ] is the last of the forwards RNN 
            #state [-1, :, : ] is the last of the backwards RNN
        
            #initial decoder hidden state is final hidden state of the forwards and backwards 
            #  encoder RNNs fed through a linear layer
            state = torch.tanh(self.fc(torch.cat((state[-2,:,:], state[-1,:,:]), dim = 1)))
            # state: [batch_size, hidden_size * dec_num_layers]
            state = torch.stack(state.split(self.hidden_size, dim=1))
            # state: [dec_num_layers, batch_size, hidden_size]
        else: # LSTM
            state, cell = state
            state = torch.tanh(self.fc(torch.cat((state[-2,:,:], state[-1,:,:]), dim = 1)))
            state = torch.stack(state.split(self.hidden_size, dim=1))
            cell = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))
            cell = torch.stack(cell.split(self.hidden_size, dim=1))
            state = (state, cell)
        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.emb_size = config.tgt_emb_size if hasattr(config, "tgt_emb_size") else config.emb_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, self.emb_size)
        self.bottleneck_size = config.bottleneck_size

        input_size = 2 * config.hidden_size + self.emb_size
        
        self.cell = config.cell
        if self.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)
        
        # bottleneck
        if self.bottleneck_size > 0:
            self.bottleneck0 = nn.Linear(config.hidden_size * 2 + config.hidden_size + self.emb_size, self.bottleneck_size)
            self.bottleneck1 = nn.Linear(self.bottleneck_size, config.tgt_vocab_size)
        else:
            self.fc_out = nn.Linear(config.hidden_size * 2 + config.hidden_size + self.emb_size,
                                    config.tgt_vocab_size)
        
        if config.attention == 'bahdanau':
            self.attention = bahdanau_attention(config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.global_emb:
            self.ge_proj1 = nn.Linear(self.emb_size, self.emb_size)
            self.ge_proj2 = nn.Linear(self.emb_size, self.emb_size)
            
    def forward(self, input, state, prediction=None, PAD_idx=None, EOS_idx=None, mask=None):
        embs = self.dropout(self.embedding(input))
        # [batch, emb size]

        if self.config.global_emb:
            if prediction is None:
                prediction = embs.new_zeros(embs.size(0), self.config.tgt_vocab_size)
                # [batch, tgt vocab size]
            probs = F.softmax(prediction / self.config.tau, dim=1)
            # [batch, tgt vocab size]
            emb_avg = torch.matmul(probs, self.embedding.weight)
            # [batch, emb size]
            H = torch.sigmoid(self.ge_proj1(embs) + self.ge_proj2(emb_avg))
            # [batch, emb size]
            embs = H * embs + (1 - H) * emb_avg
            # [batch, emb size]
        
        c_t, attn_weights = self.attention(state[-1] if self.cell == "gru" else state[0][-1])
        # c_t: [batch, 2 * hidden_size]
        
        rnn_input = torch.cat((embs, c_t), dim = 1)
        # rnn_input = [batch size, (enc hid size * 2) + emb size]
        
        output, state = self.rnn(rnn_input, state)
        # output: [batch size, dec hidden size]
        # state: [dec_num_layers, batch size, dec hidden size]
        # state for lstm: (state, cell)
        
        if self.bottleneck_size > 0:
            b0 = self.bottleneck0(torch.cat((output, c_t, embs), dim=1))
            b0 = F.dropout(torch.tanh(b0), p=self.config.bottleneck_dropout)
            prediction = self.bottleneck1(b0)
        else:
            prediction = self.fc_out(torch.cat((output, c_t, embs), dim = 1))
            
        #prediction = [batch size, tgt vocab size]
        if mask:
            mask = torch.stack(mask, dim=1).long()
            # mask: [batch, seq len of time t]
            pad, eos = None, None
            if type(PAD_idx) == int:
                pad = prediction[:,PAD_idx].clone()
            if type(EOS_idx) == int:
                eos = prediction[:,EOS_idx].clone()
                
            prediction.scatter_(dim=1, index=mask, value=-1e7)
            
            if type(PAD_idx) == int:
                prediction[torch.arange(prediction.size(0)), PAD_idx] = pad
            if type(EOS_idx) == int:
                prediction[torch.arange(prediction.size(0)), EOS_idx] = eos
        
        return prediction, state, attn_weights


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            lstm = nn.LSTMCell(input_size, hidden_size)
            self.layers.append(lstm)
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1
