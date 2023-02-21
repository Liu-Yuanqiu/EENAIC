import torch
import torch.nn as nn
from lib.config import cfg
from lib.utils import get_clip_mat

class Decoder_NA(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim=512, 
        depth=3,
        num_heads=8,
        dropout=0.1, 
        ff_dropout=0.1
    ):
        super(Decoder_NA, self).__init__()
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(depth):
            sublayer = DecoderLayer( 
                embed_dim = embed_dim, 
                num_heads = num_heads, 
                dropout = dropout, 
                ff_dropout = ff_dropout
            )
            self.layers.append(sublayer)
        
        self.word_embed = nn.Embedding(self.vocab_size, self.embed_dim)   
        self.generator = nn.Linear(self.embed_dim, self.vocab_size, bias=True)
        self.img2text = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(0.1),
        )
        self.softmax = nn.Softmax(-1)

        self.w = torch.from_numpy(get_clip_mat(17, 144)).cuda()
        self.a_embed = nn.Linear(self.embed_dim+self.embed_dim, self.embed_dim)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(self.embed_dim)

    def forward(self, encoder_out):   
        batch_size = encoder_out.shape[0]
        seq_len = cfg.MODEL.SEQ_LEN
        text_emb = self.img2text(encoder_out)
        vocab = self.word_embed(torch.arange(self.vocab_size).cuda())
        word_prob = text_emb @ vocab.t()
        word_prob = self.softmax(word_prob)
        word_emb = text_emb + word_prob @ vocab
        word_emb = torch.matmul(self.w.unsqueeze(0).to(torch.float32), word_emb)
        x = word_emb
        for layer in self.layers:
            x, x_self = layer(x, encoder_out)

        x_cat = torch.cat([x, x_self], 2)
        g = self.sigmoid(self.a_embed(x_cat))
        x = self.layernorm(x*g + x_self*(1-g))
        out = self.generator(x)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ff_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.word_attn = MultiHeadSelfAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)

        self.cross_att = MultiHeadSelfAttention(
            embed_dim = embed_dim, 
            num_heads = num_heads
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ff_layer = FeedForward(
            embed_dim = embed_dim, 
            ffn_embed_dim = embed_dim * 4, 
            relu_dropout = ff_dropout
        )
        self.layer_norm3 = torch.nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out):
        short_cut = x
        x = self.word_attn(
            q = x,
            k = x,
            v = x
        )
        x_self = x
        x = self.dropout(x)
        x = self.layer_norm1(x + short_cut)

        short_cut = x            
        x = self.cross_att(
            q = x,
            k = encoder_out,
            v = encoder_out
        )
        x = self.dropout(x)
        x = self.layer_norm2(x + short_cut)
        
        short_cut = x
        x = self.ff_layer(x)
        x = self.dropout(x)
        x = self.layer_norm3(x + short_cut)
        
        return x, x_self
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.o_linear = nn.Linear(embed_dim, embed_dim)
        
        self.softmax = nn.Softmax(-1)
    
    def forward(self, q, k, v, mask=None):
        B_, N, C = q.size()
        q = self.q_linear(q).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(B_, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, nH, L, L] or [B, nH, L, M+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
            
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = self.o_linear(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_embed_dim, relu_dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.dropout = nn.Dropout(relu_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x