import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

'''dynamic hungarian: non-empty targets match all predictions, non-empty targets match first-n predictions'''
def dynamic_hungarian_assign(decode_dist, target, EMPTY_TOKEN_IDX, assign_all_pre=False):
    '''
    :param decode_dist: (batch_size, max_lable_num, vocab_size)
    :param target: (batch_size, max_label_num)
    :return:
    '''
    batch_size, max_label_num = target.size()
    reorder_rows = torch.arange(batch_size)[..., None]
#     reorder_rows = torch.arange(batch_size).unsqueeze(1)
    
    reorder_cols = []
    for b in range(batch_size):
        reorder_col_ind = torch.arange(max_label_num).cpu().numpy()
         # find the first empty_token index
        eos_index = (target[b]==EMPTY_TOKEN_IDX).nonzero(as_tuple=True)[0]
        if eos_index.size()[0] > 0:
            eos_index = eos_index[0].item()
        else:
            eos_index = max_label_num

        if assign_all_pre:
            # non-empty targets and all predicts
            score = decode_dist[b][:, target[b, :eos_index]]
            row_ind, col_ind = linear_sum_assignment(score.detach().cpu().numpy(), maximize=True)
            last = set(reorder_col_ind) - set(row_ind)
            pad = range(eos_index,max_label_num)
            all_row_ind = list(row_ind) + list(last)
            all_col_ind = list(col_ind) + list(pad)
            for i in range(max_label_num):
                reorder_col_ind[all_row_ind[i]] = all_col_ind[i]
        else: # only assign non-empty(n) target with fisrt-n pre
            score = decode_dist[b][:eos_index, target[b, :eos_index]]
            row_ind, col_ind = linear_sum_assignment(score.detach().cpu().numpy(), maximize=True)
            reorder_col_ind[:eos_index] = col_ind
        reorder_cols.append(reorder_col_ind.reshape(1, -1))
        # total_score += sum(score[b][row_ind, col_ind])
    reorder_cols = np.concatenate(reorder_cols, axis=0)
    
    return tuple([reorder_rows, reorder_cols])

'''
the cosine distance
first dimension must be batch;
epsilon: Small value to avoid division by zero;
'''
def cos_distance(predicted_seq, ref_seq, eps=1e-8):
    # predicted_seq: [batch_size, seq_len, embedding_dim]
    # ref_seq: [batch_size, seq_len, embedding_dim]
    # batch matrix multiply, matmul is same as bmm
    # torch.matmul(x1, x2.transpose(1,2)).size()
    tmp1 = torch.bmm(predicted_seq, ref_seq.transpose(1,2))  # xTy
    tmp2 = torch.bmm(predicted_seq.norm(dim=2).unsqueeze(2), ref_seq.norm(dim=2).unsqueeze(2).transpose(1,2))  # ||x||2||y||2
    # return: [batch_size, predicted_seq_len, ref_seq_len]
    return 1 - tmp1/torch.max(tmp2,eps * torch.ones_like(tmp2))

'''
IPOT:
    Parameters
    ----------
    predicted_seq : np.ndarray (ns,)
        samples weights in the source domain
    ref_seq : np.ndarray (nt,) or np.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    C : np.ndarray (ns,nt)
        loss matrix
    beta : float, optional
        Step size of poximal point iteration
    max_iter : int, optional
        Max number of iterations
    L : int, optional
        Number of iterations for inner optimization
        
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
'''
def ipot(predicted_seq, ref_seq, C, beta=1, max_iter=50, L=1):
    # predicted_seq: [batch_size, seq_len, embedding_dim]
    # ref_seq: [batch_size, seq_len, embedding_dim]
    # C: [batch_size, predicted_seq_len, ref_seq_len]
    predicted_seq_len = predicted_seq.shape[1] # m
    ref_seq_len = ref_seq.shape[1] # n
    b = torch.ones(ref_seq.shape[0], ref_seq_len, device=predicted_seq.device)/ref_seq_len
    b = b[:,:,None] # same as b.unsqueeze(2) ; [b, n, 1]
    G = torch.exp(-C/beta) # [b, m, n]
    T = torch.ones_like(C, device=predicted_seq.device) # [b, m, n]
    for t in range(max_iter):
        Q = G * T          # [b, m, n]
        for l in range(L):
            # torch.bmm(Q,b)  is same as torch.einsum('bpr,brl->bpl',Q,b)
            a = 1 / (torch.bmm(Q, b) * predicted_seq_len)  # [b, m, 1]
            b = 1 / (torch.bmm(Q.transpose(1, 2), a) * ref_seq_len) # [b, n, 1]
        T = a * Q * b.transpose(1,2)
        # T: [batch_size, pre_seq_len, ref_seq_len]
    return T


def ipot_WD(C, beta=1, max_iter=50, L=1, use_path=True):
    m, n = C.shape
    
    u = np.ones([m,])/m
    v = np.ones([n,])/n

    P = np.ones((m,n))

    K=np.exp(-(C/beta))
  
    for outer_i in range(max_iter):

        Q = K*P
       
        if use_path == False:
            u = np.ones([m,])/m
            v = np.ones([n,])/n
           
        for i in range(L):
            u = 1/(np.matmul(Q,v)*m)
            v = 1/(np.matmul(np.transpose(Q),u)*n)
    
        P = np.expand_dims(u,axis=1)*Q*np.expand_dims(v,axis=0)

    return P

'''
OT & Bipartite matching loss
'''            
class DynamicHungarianLossAssignAll(nn.Module):
    def __init__(self, pad_index, ignore_index=True, assign_all_pre=True, empty_weight=None, lambda_ipot_E=0, ipot_E_non_empty=False, ipot_E_first_n_pre=False, auto_weight=False):
        super(DynamicHungarianLossAssignAll, self).__init__()
        
        self.assign_all_pre = assign_all_pre
        self.pad_index = pad_index
        self.empty_weight = empty_weight
        self.auto_weight = auto_weight
        self.ignore_index = ignore_index
        self.lambda_ipot_E = lambda_ipot_E
        self.ipot_E_non_empty = ipot_E_non_empty
        self.ipot_E_first_n_pre = ipot_E_first_n_pre
        self.sm = nn.Softmax(dim=2)
              
    def forward(self, predicted_seq, ref_seq, decoder_embedding=None):
        batch_size = predicted_seq.size(0)
        # hungarian negative log likelihood loss
        vocab_size = predicted_seq.size(2)
        predicted_seq = self.sm(predicted_seq)
        
        with torch.no_grad():
            reorder_index = dynamic_hungarian_assign(predicted_seq, ref_seq, self.pad_index, self.assign_all_pre)
            ref_seq_reorder = ref_seq[reorder_index]
        
        weight = None
        if self.auto_weight:
            weight = torch.ones(vocab_size, device=ref_seq.device)
            num_non_empty = (ref_seq != self.pad_index).sum().item()
            weight[self.pad_index] = num_non_empty/ (ref_seq.size(0) * ref_seq.size(1) - num_non_empty)
        elif self.empty_weight:
            weight = torch.ones(vocab_size, device=ref_seq.device)
            weight[self.pad_index] = self.empty_weight
            
        if self.ignore_index:
            ignore_index = self.pad_index
        else:
            ignore_index = -100
        
        loss = F.nll_loss(torch.log(predicted_seq.view(-1, vocab_size)), ref_seq_reorder.reshape(-1), weight=weight, ignore_index=ignore_index)
        
        # ipot embedding loss
        if self.lambda_ipot_E > 0:
            new_seq = torch.matmul(predicted_seq, decoder_embedding.weight.detach())
            with torch.no_grad():
                # ground truth embedding
                ref_seq_emb = decoder_embedding(ref_seq)
                
            ipot_embedding_loss = 0
            if not self.ipot_E_non_empty: 
                C_emb = cos_distance(new_seq, ref_seq_emb)
                with torch.no_grad():
                    T_emb = ipot(new_seq, ref_seq_emb, C_emb)
                    if self.empty_weight != None:
                        ignore_mask = ref_seq.new_zeros(ref_seq.size()).bool()
                        ignore_mask |= (ref_seq == self.pad_index)
                        ignore_mask = ignore_mask.unsqueeze(1)  # (batch_size, 1, ref_seq_len)
                        T_emb = T_emb.masked_scatter_(ignore_mask, T_emb*self.empty_weight)
                ipot_embedding_loss = torch.sum(C_emb*T_emb, (0,1,2))
            else:
                for b in range(batch_size):
                    # find the first empty_token index
                    first_pad_index = (ref_seq[b]==self.pad_index).nonzero(as_tuple=True)[0]
                    if first_pad_index.size()[0] > 0:
                        first_pad_index = first_pad_index[0].item()
                    else:
                        first_pad_index = ref_seq.size(1)
                    if self.ipot_E_first_n_pre:
                        new_seq_b = new_seq[b][:first_pad_index].unsqueeze(0) # [1, pre seq_len, emb size] non-empty assgin fisrt-n
                    else:
                        new_seq_b = new_seq[b].unsqueeze(0) # [1, pre seq_len, emb size]
                    ref_seq_emb_b = ref_seq_emb[b][:first_pad_index].unsqueeze(0) # [1, target non-empty len, emb size]
                    C_emb_b = cos_distance(new_seq_b, ref_seq_emb_b) # [1, pre seq len ,target non-empty len]
                    C_emb_b = C_emb_b.squeeze(0) # [pre seq len ,target non-empty len]
                    T_emb_b = ipot_WD(C_emb_b.detach().cpu().numpy()) # [pre seq len ,target non-empty len]
                    T_emb_b = torch.tensor(T_emb_b, device=C_emb_b.device, dtype=torch.float)
                    ipot_embedding_loss += torch.sum(C_emb_b*T_emb_b, (0,1))
                    
            ipot_embedding_loss = ipot_embedding_loss / batch_size # batch average       
            loss += (self.lambda_ipot_E * ipot_embedding_loss)
        return loss