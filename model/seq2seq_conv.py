import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rnn import rnn_encoder,StackedLSTM,StackedGRU
import random
from queue import Queue
from model.deep_lightweight_conv import dl_conv
        
class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        
    def forward(self, context, dec_hidden):
        # context: [batch_size, enc_seq_len, 2 * hidden_size]
        
        gamma_encoder = self.linear_encoder(context)
        # gamma_encoder: [batch, enc_seq_len, hidden_size]
        
        gamma_decoder = self.linear_decoder(dec_hidden).unsqueeze(1)
        # gamma_decoder: [batch, 1, hidden_size]
        
        weights = self.linear_v(torch.tanh(gamma_encoder+gamma_decoder)).squeeze(2)
        # [batch, enc_seq_len]
        weights = F.softmax(weights, dim=1)   # [batch, enc_seq_len]
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # [batch, 2 * hidden_size]

        return c_t, weights
    
class rnn_decoder(nn.Module):

    def __init__(self, config, embedding=None):
        super(rnn_decoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.emb_size = config.tgt_emb_size if hasattr(config, "tgt_emb_size") else config.emb_size
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, self.emb_size)
        self.bottleneck_size = config.bottleneck_size
        
        input_size = 4 * config.hidden_size + self.emb_size
        
        self.cell = config.cell
        if self.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)
        
        # bottleneck
        if self.bottleneck_size > 0:
            self.bottleneck0 = nn.Linear(config.hidden_size * 4 + config.hidden_size + self.emb_size, self.bottleneck_size)
            self.bottleneck1 = nn.Linear(self.bottleneck_size, config.tgt_vocab_size)
        else:
            self.fc_out = nn.Linear(config.hidden_size * 4 + config.hidden_size + self.emb_size,
                                    config.tgt_vocab_size)
        
        if config.attention == 'bahdanau':
            self.attention = bahdanau_attention(config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        if config.global_emb:
            self.ge_proj1 = nn.Linear(self.emb_size, self.emb_size)
            self.ge_proj2 = nn.Linear(self.emb_size, self.emb_size)
            
    def forward(self, context, input, state, prediction=None, PAD_idx=None, EOS_idx=None, mask=None):
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
        
        c_t, attn_weights = self.attention(context[0], state[-1] if self.cell == "gru" else state[0][-1])
        # c_t: [batch, 2 * hidden_size]
        c_t_conv, attn_weights_conv = self.attention(context[1], state[-1] if self.cell == "gru" else state[0][-1])
        # c_t: [batch, 2 * hidden_size]
        
        rnn_input = torch.cat((embs, c_t, c_t_conv), dim = 1)
        # rnn_input = [batch size, (enc hid size * 4) + emb size]
        
        output, state = self.rnn(rnn_input, state)
        # output: [batch size, dec hidden size]
        # state: [dec_num_layers, batch size, dec hidden size]
        # state for lstm: (state, cell)
        
        if self.bottleneck_size > 0:
            b0 = self.bottleneck0(torch.cat((output, c_t, c_t_conv, embs), dim=1))
            b0 = F.dropout(torch.tanh(b0), p=self.config.bottleneck_dropout)
            prediction = self.bottleneck1(b0)
        else:
            prediction = self.fc_out(torch.cat((output, c_t, c_t_conv, embs), dim = 1))
            
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

class Seq2Seq(nn.Module):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(Seq2Seq, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = rnn_encoder(config)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = rnn_decoder(config, embedding=tgt_embedding)

        self.config = config
        if hasattr(config, "stride"):
            self.dl_conv = dl_conv(2 * config.hidden_size, stride=config.stride)
        else:
            self.dl_conv = dl_conv(2 * config.hidden_size)

    def forward(self, src, src_len, dec, teacher_forcing_ratio=0):
        """
        Args:
            src: [src_len, bs]
            src_len: [bs]
            dec: [tgt_len, bs] (bos, x1, ..., xn)
        """
        tgt_len = dec.shape[0]
        
        enc_outputs, state = self.encoder(src, src_len.tolist())
        # outputs: (seq_len, batch_size, 2 * hidden_size)
        context = enc_outputs.transpose(0, 1)
        # context: (batch_size, seq_len, 2 * hidden_size)
        
        context_conv = self.dl_conv(context.transpose(1, 2))
        context_conv = context_conv.transpose(1, 2)
            
        outputs = []
        output = None
        
        # first input to the decoder is the <sos> token.
        input = dec[0]

        for t in range(1, tgt_len):
            output, state, _= self.decoder((context, context_conv), input, state, output)
            # output: [batch size, tgt vocab size]
            outputs.append(output)
            
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # [batch size]
            
            # update input : use ground_truth when teacher_force 
            input = dec[t] if teacher_force else top1
            
        outputs = torch.stack(outputs)

        return outputs
    
    def sample(self, src, src_len, SOS_idx, PAD_idx, EOS_idx, masked_softmax=False):
        """
        Args:
            src: [src_len, bs]
            src_len: [bs]
        """
                
        batch_size = src.size(1)
        
        enc_outputs, state = self.encoder(src, src_len.tolist())
        # outputs: (seq_len, batch_size, 2 * hidden_size)
        context = enc_outputs.transpose(0, 1)
        # context: (batch_size, seq_len, 2 * hidden_size)
        
        context_conv = self.dl_conv(context.transpose(1, 2))
        context_conv = context_conv.transpose(1, 2)
           
        outputs, predicted = [],[]
        output = None
        
        # first input to the decoder is the <sos> token.
        input = torch.tensor([SOS_idx]*batch_size, device=src.device)

        for t in range(self.config.max_time_step):
            output, state, _= self.decoder((context, context_conv), input, state, output, PAD_idx, EOS_idx, predicted if masked_softmax else None)
            # output: [batch size, tgt vocab size]
            outputs.append(output)
            
            # get the highest predicted token from our predictions.
            input = output.argmax(1)
            # [batch size]
            
            # for softmax mask
            predicted += [input]
            # seq_len of t * [batch size]
            
        outputs = torch.stack(outputs)

        return outputs