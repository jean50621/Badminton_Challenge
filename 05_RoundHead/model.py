import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import join as pjoin

# TransFG code:
# reference https://github.com/TACJu/TransFG/blob/master/models/modeling.py
# thanks !!

"""
B: batch size
L: sequence length
C: channel
"""

def swish(x):
    return x * torch.sigmoid(x)

class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 num_att_heads: int,
                 hidden_size: int,
                 att_dropout: float,
                 proj_dropout:float) -> None:
        super(MultiHeadAttention, self).__init__()

        self.num_att_heads = num_att_heads
        self.hidden_size = hidden_size
        self.att_head_size = int(hidden_size / num_att_heads)
        self.all_head_size = num_att_heads * self.att_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.att_dropout = nn.Dropout(p=att_dropout)
        self.proj_dropout = nn.Dropout(p=proj_dropout)

    def forward(self, x):
        """
        x.size() == [B, L, C]
        """
        B, L, C = x.size()

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # B, L, C --> B, n_heads, L, head_size
        q = q.view(B, L, self.num_att_heads, self.att_head_size).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, L, self.num_att_heads, self.att_head_size).permute(0, 2, 3, 1).contiguous()
        v = v.view(B, L, self.num_att_heads, self.att_head_size).permute(0, 2, 1, 3).contiguous()

        # score size: B, n_heads, L, L
        scores = torch.matmul(q, k)
        scores = scores / self.att_head_size**0.5
        probs = torch.softmax(scores, dim=-1)

        """ Self-Attention Map that TransFG used """
        weights = probs
        probs = self.att_dropout(probs)

        # context 
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous() # -> B, L, n_heads, head_size
        context = context.view(B, L, -1)

        out = self.proj(context)
        out = self.proj_dropout(out)

        return out, weights


class FeedForwardNet(nn.Module):

    def __init__(self, 
                 hidden_size: int, 
                 dim_feedforward: int,
                 dropout: float) -> None:
        super(FeedForwardNet, self).__init__()
        """
        usually set dim_feedforward == 4 * hidden_size
        """ 

        self.fc1 = nn.Linear(hidden_size, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, hidden_size)
        self.act = F.gelu
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    
    def __init__(self, 
                 length: int = 80,
                 out_channels: int = 64,
                 dropout: float = 0.0):
        super(Embeddings, self).__init__()

        # self.length = length
        _length = 132
        _out = int(64)
        self.ball_norm = nn.LayerNorm([_length, 2])
        self.ball_emb = nn.Linear(2, _out * 3)
        self.A_norm = nn.LayerNorm([_length, 4])
        self.A_emb = nn.Linear(4, _out)
        self.B_norm = nn.LayerNorm([_length, 4])
        self.B_emb = nn.Linear(4, _out)
        self.net_norm = nn.LayerNorm([_length, 4])
        self.net_emb = nn.Linear(4, _out)
        self.court_norm = nn.LayerNorm([_length, 4])
        self.court_emb = nn.Linear(4, _out)

        self.pos_embedding = nn.Parameter(torch.zeros(1, length + 1, out_channels))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, ball, A, B, net, court):

        """
        ball: B, L, 2
        A: B, L, 4
        ...
        """

        batch_size = ball.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        ball = self.ball_norm(ball)
        ball = self.ball_emb(ball)
        
        A = self.A_norm(A)
        A = self.A_emb(A)
        
        B = self.B_norm(B)
        B = self.B_emb(B)

        net = self.net_norm(net)
        net = self.net_emb(net)

        court = self.court_norm(court)
        court = self.court_emb(court)

        x = torch.cat((ball, A, B, net, court), dim=2)

        x = torch.cat((cls_tokens, x), dim=1) # batch_size L + 1, C

        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):

    def __init__(self, 
                 d_model: int=512, 
                 nhead: int=8, 
                 dim_feedforward: int=2048, 
                 ffn_dropout: float=0.0,
                 att_dropout: float=0.0,
                 proj_dropout: float=0.0):
        super(TransformerLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.ffn_dropout = ffn_dropout
        self.att_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = FeedForwardNet(d_model, dim_feedforward, ffn_dropout)
        self.multiheads_att = MultiHeadAttention(nhead, d_model, att_dropout, proj_dropout)

    def forward(self, x):
        h = x
        x = self.att_norm(x)
        x, weights = self.multiheads_att(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):

    def __init__(self, 
                 num_layers: int = 12,
                 d_model: int=512, 
                 nhead: int=8, 
                 dim_feedforward: int=2048, 
                 ffn_dropout: float=0.0,
                 att_dropout: float=0.0,
                 proj_dropout: float=0.0):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module("TransformerLayer_{}".format(i), 
                             TransformerLayer(d_model=d_model, 
                                              nhead=nhead, 
                                              dim_feedforward=dim_feedforward, 
                                              ffn_dropout=ffn_dropout,
                                              att_dropout=att_dropout,
                                              proj_dropout=proj_dropout))

        self.encoder_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        
        for i in range(self.num_layers):
            x, weights = getattr(self, "TransformerLayer_{}".format(i))(x)

        encode = self.encoder_norm(x)   

        return encode

class Transformer(nn.Module):

    def __init__(self, 
                 length: int = 80,
                 d_model: int = 64,
                 dropout: float = 0.0,
                 num_layers: int = 8,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 ffn_dropout: float=0.0,
                 att_dropout: float=0.0,
                 proj_dropout: float=0.0):

        super(Transformer, self).__init__()
        self.embeddings = Embeddings(length, d_model, dropout)
        self.encoder = Encoder(num_layers, d_model, nhead, dim_feedforward, ffn_dropout, att_dropout, proj_dropout)

    def forward(self, ball, A, B, net, court):
        x = self.embeddings(ball, A, B, net, court)
        x = self.encoder(x)
        return x


class BT_Transformer(nn.Module):
    
    def __init__(self, 
                 length: int = 80,
                 d_model: int = 64,
                 dropout: float = 0.0,
                 num_layers: int = 8,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 ffn_dropout: float=0.0,
                 att_dropout: float=0.0,
                 proj_dropout: float=0.0):
        super(BT_Transformer, self).__init__()
        
        self.transformer = Transformer(length, d_model, dropout, num_layers, \
                                       nhead, dim_feedforward, ffn_dropout, att_dropout, proj_dropout)


    def forward(self, ball, A, B, net, court):
        """
        x : [B, C, H, W]
        """
        part_tokens = self.transformer(ball, A, B, net, court)

        return part_tokens[:, 0]



class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.model_trans = BT_Transformer(length = 132,
                                          d_model = 448,
                                          dropout = 0.0,
                                          num_layers = 12,
                                          nhead = 16,
                                          dim_feedforward = 2048,
                                          ffn_dropout = 0.05,
                                          att_dropout = 0.05,
                                          proj_dropout = 0.05)
        self.model_trans_projector = nn.Linear(448, 512)
        
        model_name = 'x3d_s'
        self.model_c3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
        # print(self.model)
        self.model_c3d.blocks[5].pool.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.model_c3d.blocks[5].proj = nn.Identity()
        self.model_c3d.blocks[5].output_proj = nn.Identity()
        self.model_c3d_projector = nn.Linear(2048, 512)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, video, ball, A, B, net, court):
        video_hs = self.model_c3d(video)
        # print(video_hs.size()) # [4, 2048]
        video_hs = self.model_c3d_projector(video_hs)

        seq_hs = self.model_trans(ball, A, B, net, court)
        # print(seq_hs.size()) # [4, 448]
        seq_hs = self.model_trans_projector(seq_hs)

        x = torch.cat((video_hs, seq_hs), dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = \
        BT_Transformer(num_classes = 36,
                       length = 81,
                       d_model = 384,
                       dropout = 0.0,
                       num_layers = 12,
                       nhead = 16,
                       dim_feedforward = 2048,
                       ffn_dropout = 0.05,
                       att_dropout = 0.05,
                       proj_dropout = 0.05)

    ball = torch.randn(2, 81, 2)
    A = torch.randn(2, 81, 4)
    B = torch.randn(2, 81, 4)
    net = torch.randn(2, 81, 4)
    court = torch.randn(2, 81, 4)
    out = model(ball, A, B, net, court)
    print(out.size())

