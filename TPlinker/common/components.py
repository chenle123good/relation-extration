from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim = 0, center = True, scale = True, epsilon = None, conditional = False,
                 hidden_units = None, hidden_activation = 'linear', hidden_initializer = 'xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features = self.cond_dim, out_features = self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢? 
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            
            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
    
class HandshakingKernel(nn.Module): # 共享编码
    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, 
                           hidden_size, 
                           num_layers = 1, 
                           bidirectional = False, 
                           batch_first = True)
     
    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type = "lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :] # 一个个遍历行，当是第一行时，ind是0，hidden_each_step维度是[batch,1,hidden_size]代表整句话第一个word的编码，为了进行拼接147行repeat_hiddens在第二个维度进行了复制，维度变成了[batch,5,hidden_size],相当于将当前单词编码复制了5份，visible_hiddens维度就是[batch,5,hidden_size]，是从当前单词往后（包括自身）各个单词的编码，现在要计算得到上三角第一行的编码，即150行的shaking_hiddens，将当前单词和其后的各个单词的编码进行concat维度是[batch,5,hidden_size*2],然后151行又过了一个MLP层，转化为了shaking_hiddens [batch,5,hidden_size]
            visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)  # 从当前取到最后，优化为上三角矩阵
            
            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)

            shaking_hiddens_list.append(shaking_hiddens) # 就是记录上三角一行行的编码
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1) # 公共编码成上三角矩阵
        return long_shaking_hiddens