import numpy as np
import torch.nn as nn
import torch
from gensim.models import KeyedVectors

glove_path = "data/glove.840B.300d.gensim"

def src_embedding_glove(device, SRC, freeze=False):
    emb_size = 300 
    w2v_model = KeyedVectors.load(glove_path)
    src_embedding = np.random.uniform(-0.1, 0.1, (len(SRC.vocab), emb_size))
    for i in range(SRC.vocab.__len__()):
        word = SRC.vocab.itos[i]
        if word in w2v_model:
            src_embedding[i] = w2v_model[word]
    pad_idx = SRC.vocab.stoi[SRC.pad_token]
    src_embedding[pad_idx] = np.zeros(emb_size)
    src_embedding = nn.Embedding.from_pretrained(torch.tensor(src_embedding, device=device, dtype=torch.float), freeze=freeze)
    return src_embedding

def tgt_embedding_glove(device, TGT, freeze=False):
    emb_size = 300 
    w2v_model = KeyedVectors.load(glove_path)
    tgt_embedding = np.random.uniform(-0.1, 0.1, (len(TGT.vocab), emb_size))
    for i in range(TGT.vocab.__len__()):
        label = TGT.vocab.itos[i]
        words_emb = []
        for word in label.split("_"):
            if word in w2v_model:
                words_emb.append(w2v_model[word])
            else:
                words_emb.append(np.random.uniform(-0.1, 0.1, emb_size))
        tgt_embedding[i] = np.mean(np.asarray(words_emb), axis=0)
    pad_idx = TGT.vocab.stoi[TGT.pad_token]
    tgt_embedding[pad_idx] = np.zeros(emb_size)
    tgt_embedding = nn.Embedding.from_pretrained(torch.tensor(tgt_embedding, device=device, dtype=torch.float), freeze=freeze)
    return tgt_embedding