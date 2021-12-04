# -*- coding: utf-8 -*-

from .BasicModule import BasicModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCNN_ATT(BasicModule):
    '''
    Lin 2016 Att PCNN
    '''
    def __init__(self, opt):
        super(PCNN_ATT, self).__init__()

        self.opt = opt
        self.model_name = 'PCNN_ATT'
        self.test_scale_p = 0.5

        self.word_embs = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)
        self.pos2_embs = nn.Embedding(self.opt.pos_size, self.opt.pos_dim)

        all_filter_num = self.opt.filters_num * len(self.opt.filters)

        rel_dim = all_filter_num

        if self.opt.use_pcnn:
            rel_dim = all_filter_num * 3
            masks = torch.LongTensor(([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
            if self.opt.use_gpu:
                masks = masks.cuda()
            self.mask_embedding = nn.Embedding(4, 3)
            self.mask_embedding.weight.data.copy_(masks)
            self.mask_embedding.weight.requires_grad = False

        self.rel_embs = nn.Parameter(torch.randn(self.opt.rel_num, rel_dim))
        self.rel_bias = nn.Parameter(torch.randn(self.opt.rel_num))

        # the Relation-Specific Attention Diagonal Matrix
        # self.att_w = nn.ParameterList([nn.Parameter(torch.eye(rel_dim)) for _ in range(self.opt.rel_num)])

        # Conv filter width
        feature_dim = self.opt.word_dim + self.opt.pos_dim * 2

        # option for multi size filter
        # here is only a kind of filter with height = 3
        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0)) for k in self.opt.filters])
        self.dropout = nn.Dropout(self.opt.drop_out)

        self.init_model_weight()
        self.init_word_emb()

    def init_model_weight(self):
        '''
        use xavier to init
        '''
        nn.init.xavier_uniform(self.rel_embs)
        nn.init.uniform(self.rel_bias)
        for conv in self.convs:
            nn.init.xavier_uniform(conv.weight)
            nn.init.uniform(conv.bias)

    def init_word_emb(self):

        def p_2norm(path):
            v = torch.from_numpy(np.load(path))
            if self.opt.norm_emb:
                v = torch.div(v, v.norm(2, 1).unsqueeze(1))
                v[v != v] = 0.0
            return v

        w2v = p_2norm(self.opt.w2v_path)
        p1_2v = p_2norm(self.opt.p1_2v_path)
        p2_2v = p_2norm(self.opt.p2_2v_path)

        if self.opt.use_gpu:
            self.word_embs.weight.data.copy_(w2v.cuda())
            self.pos1_embs.weight.data.copy_(p1_2v.cuda())
            self.pos2_embs.weight.data.copy_(p2_2v.cuda())
        else:
            self.pos1_embs.weight.data.copy_(p1_2v)
            self.pos2_embs.weight.data.copy_(p2_2v)
            self.word_embs.weight.data.copy_(w2v)

    def init_int_constant(self, num):
        '''
        a util function for generating a LongTensor Variable
        '''
        if self.opt.use_gpu:
            return Variable(torch.LongTensor([num]).cuda())
        else:
            return Variable(torch.LongTensor([num]))

    def mask_piece_pooling(self, x, mask):
        '''
        refer: https://github.com/thunlp/OpenNRE
        A fast piecewise pooling using mask
        '''
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)
        masks = self.mask_embedding(mask).unsqueeze(-2) * 100
        x = masks + x
        x = torch.max(x, 1)[0] - 100
        return x.view(-1, x.size(1) * x.size(2))

    def piece_max_pooling(self, x, insPool):
        '''
        piecewise pool into 3 segements
        x: the batch data
        insPool: the batch Pool
        '''
        split_batch_x = torch.split(x, 1, 0)
        split_pool = torch.split(insPool, 1, 0)
        batch_res = []

        for i in range(len(split_pool)):
            ins = split_batch_x[i].squeeze()                                    # all_filter_num * max_len
            pool = split_pool[i].squeeze().data                                 # 2
            seg_1 = ins[:, :pool[0]].max(1)[0].unsqueeze(1)                     # all_filter_num * 1
            seg_2 = ins[:, pool[0]: pool[1]].max(1)[0].unsqueeze(1)             # all_filter_num * 1
            seg_3 = ins[:, pool[1]:].max(1)[0].unsqueeze(1)
            piece_max_pool = torch.cat([seg_1, seg_2, seg_3], 1).view(1, -1)    # 1 * 3all_filter_num
            batch_res.append(piece_max_pool)

        out = torch.cat(batch_res, 0)
        assert out.size(1) == 3 * self.opt.filters_num
        return out

    def forward(self, x, label=None):

        # get all sentences embedding in all bags of one batch
        self.bags_feature = self.get_bags_feature(x)

        if label is None:
            # for test
            assert self.training is False
            return self.test(x)
        else:
            # for train
            assert self.training is True
            return self.fit(x, label)

    def fit(self, x, label):
        '''
        train process
        '''
        x = self.get_batch_feature(label)               # batch_size * sentence_feature_num
        x = self.dropout(x)
        out = x.mm(self.rel_embs.t()) + self.rel_bias     # o = Ms + d (formual 10 in paper)

        if self.opt.use_gpu:
            v_label = torch.LongTensor(label).cuda()
        else:
            v_label = torch.LongTensor(label)
        ce_loss = F.cross_entropy(out, Variable(v_label))
        return ce_loss

    def test(self, x):
        '''
        test process
        '''
        pre_y = []
        for label in range(0, self.opt.rel_num):
            labels = [label for _ in range(len(x))]                 # generate the batch labels
            bags_feature = self.get_batch_feature(labels)
            out = self.test_scale_p * bags_feature.mm(self.rel_embs.t()) + self.rel_bias
            # out = F.softmax(out, 1)
            # pre_y.append(out[:, label])
            pre_y.append(out.unsqueeze(1))

        # return pre_y
        res = torch.cat(pre_y, 1).max(1)[0]
        return F.softmax(res, 1).t()

    def get_batch_feature(self, labels):
        '''
        Using Attention to get all bags embedding in a batch
        '''
        batch_feature = []

        for bag_embs, label in zip(self.bags_feature, labels):
            # calculate the weight: xAr or xr
            alpha = bag_embs.mm(self.rel_embs[label].view(-1, 1))
            # alpha = bag_embs.mm(self.att_w[label]).mm(self.rel_embs[label].view(-1, 1))
            bag_embs = bag_embs * F.softmax(alpha, 0)
            bag_vec = torch.sum(bag_embs, 0)
            batch_feature.append(bag_vec.unsqueeze(0))

        return torch.cat(batch_feature, 0)

    def get_bags_feature(self, bags):
        '''
        get all bags embedding in one batch before Attention
        '''
        bags_feature = []
        for bag in bags:
            if self.opt.use_gpu:
                data = map(lambda x: Variable(torch.LongTensor(x).cuda()), bag)
            else:
                data = map(lambda x: Variable(torch.LongTensor(x)), bag)

            bag_embs = self.get_ins_emb(data)                                   # get all instances embedding in one bag
            bags_feature.append(bag_embs)

        return bags_feature

    def get_ins_emb(self, x):
        '''
        x: all instance in a Bag
        '''
        insEnt, _, insX, insPFs, insPool, mask = x
        insPF1, insPF2 = [i.squeeze(1) for i in torch.split(insPFs, 1, 1)]

        word_emb = self.word_embs(insX)
        pf1_emb = self.pos1_embs(insPF1)
        pf2_emb = self.pos2_embs(insPF2)

        x = torch.cat([word_emb, pf1_emb, pf2_emb], 2)                          # insNum * 1 * maxLen * (word_dim + 2pos_dim)
        x = x.unsqueeze(1)                                                      # insNum * 1 * maxLen * (word_dim + 2pos_dim)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [self.mask_piece_pooling(i, mask) for i in x]
        # x = [self.piece_max_pooling(i, insPool) for i in x]
        x = torch.cat(x, 1).tanh()
        return x
