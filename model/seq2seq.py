import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rnn import rnn_encoder,rnn_decoder
import random
from queue import Queue

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
        enc_outputs = enc_outputs.transpose(0, 1)
        # outputs: (batch_size, seq_len, 2 * hidden_size)
            
        self.decoder.attention.init_context(context=enc_outputs)

        outputs = []
        output = None
        
        # first input to the decoder is the <sos> token.
        input = dec[0]

        for t in range(1, tgt_len):
            output, state, _= self.decoder(input, state, output)
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
        enc_outputs = enc_outputs.transpose(0, 1)
        # outputs: (batch_size, seq_len, 2 * hidden_size)
            
        self.decoder.attention.init_context(context=enc_outputs)

        outputs, predicted = [],[]
        output = None
        
        # first input to the decoder is the <sos> token.
        input = torch.tensor([SOS_idx]*batch_size, device=src.device)

        for t in range(self.config.max_time_step):
            output, state, _= self.decoder(input, state, output, PAD_idx, EOS_idx, predicted if masked_softmax else None)
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