import math
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg
from torch.nn.utils.weight_norm import weight_norm

def activation(act):
    if act == 'RELU':
        return nn.ReLU(inplace=True)
    elif act == 'TANH':
        return nn.Tanh()
    elif act == 'GLU':
        return nn.GLU()
    elif act == 'ELU':
        return nn.ELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'CELU':
        return nn.CELU(cfg.MODEL.BILINEAR.ELU_ALPHA, inplace=True)
    elif act == 'GELU':
        return nn.GELU()
    else:
        return nn.Identity()

def expand_tensor(tensor, size, dim=1):
    if size == 1 or tensor is None:
        return tensor
    tensor = tensor.unsqueeze(dim)
    tensor = tensor.expand(list(tensor.shape[:dim]) + [size] + list(tensor.shape[dim+1:])).contiguous()
    tensor = tensor.view(list(tensor.shape[:dim-1]) + [-1] + list(tensor.shape[dim+1:]))
    return tensor
    
def get_clip_mat(length, size):
    w = np.zeros((length, size), dtype = float)
    t = 0.3
    for j in range(length):
        for i in range(size):
            w[j][i] = np.exp(-(j-i*length/size)**2/t)
    w = w/144.0
    return w

def expand_numpy(x, size=cfg.DATA_LOADER.SEQ_PER_IMG):
    if cfg.DATA_LOADER.SEQ_PER_IMG == 1:
        return x
    x = x.reshape((-1, 1))
    x = np.repeat(x, size, axis=1)
    x = x.reshape((-1))
    return x

def load_ids(path):
    with open(path, 'r') as fid:
        lines = [int(line.strip()) for line in fid]
    return lines

def load_lines(path):
    with open(path, 'r') as fid:
        lines = [line.strip() for line in fid]
    return lines

def load_vocab(path):
    vocab = ['.']
    with open(path, 'r') as fid:
        for line in fid:
            vocab.append(line.strip())
    return vocab

# torch.nn.utils.clip_grad_norm
# https://github.com/pytorch/examples/blob/master/word_language_model/main.py#L84-L91
# torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
def clip_gradient(optimizer, model, grad_clip_type, grad_clip):
    if grad_clip_type == 'Clamp':
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.requires_grad == True:
                    param.grad.data.clamp_(-grad_clip, grad_clip)
    elif grad_clip_type == 'Norm':
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif grad_clip_type == 'None':
        pass
    else:
        raise NotImplementedError
        
def fill_list(_list, fill, length):
    _len = len(_list)
    if _len > length:
        return _list[:length]
    else:
        _fill = [fill for i in range(length-_len)]
        return _list + _fill

def decode_index(seq):
    if len(seq.shape)>1:
        batch_size = seq.shape[0]
        len1 = seq.shape[1]
        out = []
        for i in range(batch_size):
            out_i = []
            out_i.append(0)
            for j in range(len1):
                if j != 0:
                    out_i.append(j)
                else:
                    out_i.append(0)
                    break
            out_i = fill_list(out_i, 0, cfg.MODEL.SEQ_LEN)
            out.append(out_i)
        return out
    else:
        out_i = []
        out_i.append(0)
        for j in range(len):
            if j != 0:
                out_i.append(j)
            else:
                out_i.append(0)
                break
        out_i = fill_list(out_i, 0, cfg.MODEL.SEQ_LEN)
        return out_i

def decode_sequence(vocab, seq):
    N, T = seq.size()
    sents = []
    for n in range(N):
        words = []
        for t in range(T):
            ix = seq[n, t]
            if ix == 0:
                break
            words.append(vocab[ix])
        sent = ' '.join(words)
        sents.append(sent)
    return sents

def clip_chongfu(sents):
    caps = []
    for sent in sents:
        words = sent.split(" ")
        for i in range(len(words)-1, 0, -1):
            if words[i] == words[i-1]:
                del words[i]
        cap = ' '.join(words)
        caps.append(cap)
    return caps

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float(-1e9)).type_as(t)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count