# OTSeq2Set

---

OTSeq2Set: An Optimal Transport Enhanced Sequence-to-Set Model for Extreme Multi-label Text Classification



## Dependency

---

torch==1.9.0 

torchtext==0.10.0 



## Datasets

---



OTSeq2set uses the same dataset as AttentionXML, please download each dataset from the following links.

* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)

The gensim format GloVe embedding (840B,300d) is provided by AttentionXML [here](https://drive.google.com/file/d/10w_HuLklGc8GA_FtUSdnHT8Yo1mxYziP/view?usp=sharing).

For Wiki10-31K,AmazonCat-13K, the label vocabulary is downloaded from [The Extreme Classification Repository](http://manikvarma.org/downloads/XC/XMLRepository.html#provenance)

### or

We compress four datasets with label vocabulary and Glove embedding [here](https://drive.google.com/file/d/1CH2C4dFyx6MEhOr5pezDpsRZNqJglPT9/view?usp=sharing).

The structure of the dataset should be:

```
OTSeq2Set
 |-- config
 |-- data                          
 |    |-- Eurlex              
 |    |-- AmazonCat-13K
 |    |-- Amazon-670K
 |    |-- Wiki10-31K       
 |    |-- glove.840B.300d.gensim
 |    |-- glove.840B.300d.gensim.vectors.npy
 |-- OTSet2Set.ipynb       
```



## Config

---

File **config/OTSeq2Set.json** contains the configuration of OTSeq2Set which the results are shown in the paper.

**config/baselines.json** contains the configuration of baseline models.

Description of configuration:

* **dl_conv** : use light weight convolution or not
* **lambda_embedding**: The parameter  **lambda** of semantic optimal transport distance
* **finish** :  whether the model is trained or not, **needs set to true if you don't want to train this model**

## Train

---

Run **OTSeq2Set.ipynb**

