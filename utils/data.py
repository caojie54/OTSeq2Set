from torchtext.legacy.data import Field,Dataset,Iterator,Example,BucketIterator
import random
import os
import gc
import pandas as pd
import numpy as np
from collections import Counter

def process_src(lines, max_src_len=500, min_src_len=52, pad_token="<pad>"):
    lines = [line.split()[:max_src_len] for line in lines]
    new_lines = []
    for words in lines:
        src_len = len(words)
        if src_len < min_src_len:
            new_lines.append(' '.join(words + (min_src_len-src_len)*[pad_token]))
        else:
            new_lines.append(' '.join(words))
    return new_lines

# EUR_Lex
def processing_data_common(train_src, test_src, train_tgt, test_tgt, max_src_len, device, batch_first=False, batch_size=64, include_lengths=False, 
                           for_CE=True, valid_split=0.1, fix_label_length=None, tgt_sort=False, test_model=False):
    
    if test_model:
        train_src = train_src[:1000]
        
    train_src = process_src(train_src, max_src_len)
    test_src = process_src(test_src, max_src_len)
    
    if tgt_sort:
        counter = Counter()
        tgt = train_tgt+test_tgt
        for line in tgt:
            counter.update(line.split())
        train_tgt = [' '.join(sorted(line.split(), key=lambda x:counter[x], reverse=True)) for line in train_tgt]
        test_tgt = [' '.join(sorted(line.split(), key=lambda x:counter[x], reverse=True)) for line in test_tgt]
        
    train_data = list(zip(train_src, train_tgt))
    
    del train_src,train_tgt
    gc.collect()
    
    test_data = list(zip(test_src, test_tgt))
    
    del test_src,test_tgt
    gc.collect()
    
    SRC = Field(include_lengths=include_lengths, batch_first=batch_first)
        
    if for_CE:
        TGT = Field(init_token='<sos>', eos_token="<eos>", batch_first=batch_first)
    else:
        if fix_label_length and fix_label_length > 0:
            TGT = Field(init_token='<sos>', fix_length=fix_label_length, batch_first=batch_first) # fix_length include SOS
        else:
            TGT = Field(init_token='<sos>', batch_first=batch_first)

    FIELDS = [('src', SRC), ('tgt', TGT)]

    train_examples = list(map(lambda x: Example.fromlist(x, fields=FIELDS), train_data))
    
    del train_data
    gc.collect()
    
    test_examples = list(map(lambda x: Example.fromlist(x, fields=FIELDS), test_data))

    del test_data
    gc.collect()
    
    train_dataset = Dataset(train_examples, fields=FIELDS)
    test_dataset = Dataset(test_examples, fields=FIELDS)

    total_examples = train_examples + test_examples 
    
    del train_examples, test_examples
    gc.collect()
    
    total_dataset = Dataset(total_examples, fields=FIELDS)
    
    del total_examples
    gc.collect()
    
    SRC.build_vocab(total_dataset, max_size=50000)
    TGT.build_vocab(total_dataset)
    
    del total_dataset
    gc.collect()
    
    valid_iter = None
    if valid_split > 0 and valid_split < 1:
        train_dataset, valid_dataset = train_dataset.split(split_ratio=[1-valid_split, valid_split])
        train_iter, valid_iter, test_iter = BucketIterator.splits(
            (train_dataset,valid_dataset,test_dataset),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.src), 
            sort_within_batch=True,
            repeat=False
        )
    else:
        train_iter, test_iter = BucketIterator.splits(
            (train_dataset,test_dataset),
            batch_size=batch_size,
            device=device,
            sort_key=lambda x: len(x.src), 
            sort_within_batch=True,
            repeat=False
        )
    return train_iter,valid_iter,test_iter,SRC,TGT

def delete_short(src, tgt):
    new_src, new_tgt = [], []
    for i, line in enumerate(src):
        if len(line.split())>=1:
            new_src.append(line)
            new_tgt.append(tgt[i])
    return new_src, new_tgt

#wiki 31K
def processing_data_wiki31k(device, max_src_len=2000, **kw):
    data_path = "data/Wiki10-31K"

    with open(f"{data_path}/train_labels.txt", 'r') as f:
        train_tgt_ids = f.readlines()

    with open(f"{data_path}/test_labels.txt", 'r') as f:
        test_tgt_ids = f.readlines()

    with open(f"{data_path}/Y.txt", 'r') as f:
        tgt_vocab = f.readlines()

    with open(f"{data_path}/train_raw_texts.txt", 'r') as f:
        train_src = f.readlines()

    with open(f"{data_path}/test_raw_texts.txt", 'r') as f:
        test_src = f.readlines()

    tgt_vocab = [label.strip() for label in tgt_vocab]

    train_tgt = [' '.join([tgt_vocab[int(i)] for i in line.split()]) for line in train_tgt_ids]

    test_tgt = [' '.join([tgt_vocab[int(i)] for i in line.split()]) for line in test_tgt_ids]
    
    train_src, train_tgt = delete_short(train_src, train_tgt)
    
    test_src, test_tgt = delete_short(test_src, test_tgt)
    
    tgt_lengths = [len(x.split()) for x in train_tgt+test_tgt]

    max_tgt_lengths = max(tgt_lengths)
    
    return processing_data_common(train_src, test_src, train_tgt, test_tgt, max_src_len, device, fix_label_length=max_tgt_lengths+1, **kw)

#wiki 31K
def processing_data_Amazon670k(device, max_src_len=500, **kw):
    data_path = "data/Amazon-670K"

    with open(f"{data_path}/train_labels.txt", 'r') as f:
        train_tgt = f.readlines()

    with open(f"{data_path}/test_labels.txt", 'r') as f:
        test_tgt = f.readlines()

    with open(f"{data_path}/train_raw_texts.txt", 'r') as f:
        train_src = f.readlines()

    with open(f"{data_path}/test_raw_texts.txt", 'r') as f:
        test_src = f.readlines()
    
    tgt_lengths = [len(x.split()) for x in train_tgt+test_tgt]

    max_tgt_lengths = max(tgt_lengths)
    
    assert max_tgt_lengths == 7
    
    assert len(train_tgt) == len(train_src)
    
    assert len(test_tgt) == len(test_src)
    
    return processing_data_common(train_src, test_src, train_tgt, test_tgt, max_src_len, device, fix_label_length=max_tgt_lengths+1, **kw)
  
    
# eur-lex
def processing_data_EUR_Lex(device, max_src_len=1000, **kw):
    
    data_path = "./data/EUR-Lex/"

    with open(f"{data_path}train_texts.txt", 'r') as f:
        train_src = f.readlines()
    with open(f"{data_path}test_texts.txt", 'r') as f:
        test_src = f.readlines()
    with open(f"{data_path}train_labels.txt", 'r') as f:
        train_tgt = f.readlines()
    with open(f"{data_path}test_labels.txt", 'r') as f:
        test_tgt = f.readlines()
    
    tgt_lengths = [len(x.split()) for x in train_tgt+test_tgt]

    max_tgt_lengths = max(tgt_lengths)
    
    assert max_tgt_lengths == 24
    
    assert len(train_tgt) == len(train_src)
    
    assert len(test_tgt) == len(test_src)
    
    return processing_data_common(train_src, test_src, train_tgt, test_tgt, max_src_len, device, fix_label_length=max_tgt_lengths+1, **kw)


# AmazonCat-13K
def processing_data_AmazonCat13k(device, max_src_len=500, **kw):
    data_path = "data/AmazonCat-13K"

    with open(f"{data_path}/train_labels.txt", 'r') as f:
        train_tgt_ids = f.readlines()

    with open(f"{data_path}/test_labels.txt", 'r') as f:
        test_tgt_ids = f.readlines()

    with open(f"{data_path}/Yf.txt", 'r', encoding='latin-1') as f:
        tgt_vocab = f.readlines()

    with open(f"{data_path}/train_raw_texts.txt", 'r') as f:
        train_src = f.readlines()

    with open(f"{data_path}/test_raw_texts.txt", 'r') as f:
        test_src = f.readlines()

    tgt_vocab = ['_'.join(label.strip().replace("&", ' ').split()) for label in tgt_vocab]

    train_tgt = [' '.join([tgt_vocab[int(i)] for i in line.split()]) for line in train_tgt_ids]

    test_tgt = [' '.join([tgt_vocab[int(i)] for i in line.split()]) for line in test_tgt_ids]

    tgt_lengths = [len(x.split()) for x in train_tgt+test_tgt]

    max_tgt_lengths = max(tgt_lengths)

    assert max_tgt_lengths == 57

    assert len(train_tgt) == len(train_src)

    assert len(test_tgt) == len(test_src)
    
    return processing_data_common(train_src, test_src, train_tgt, test_tgt, max_src_len, device, fix_label_length=max_tgt_lengths+1, **kw)