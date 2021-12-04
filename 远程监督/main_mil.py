# -*- coding: utf-8 -*-

from config import opt
import models
import dataset
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils import save_pr, now, eval_metric


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def test(**kwargs):
    pass


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(**kwargs):

    setup_seed(opt.seed)

    kwargs.update({'model': 'PCNN_ONE'})
    opt.parse(kwargs)

    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # 执行指定文件夹下的_init_文件，执行相应导入文件——getattr(文件夹，文件夹中的文件)
    # torch.manual_seed(opt.seed)
    model = getattr(models, 'PCNN_ONE')(opt) # 执行命令行，出初始化相应参数，此处为model参数，也就是初始化模型
    if opt.use_gpu:
        # torch.cuda.manual_seed_all(opt.seed)
        model.cuda()
        # parallel
        #  model = nn.DataParallel(model)

    # loading data
    DataModel = getattr(dataset, opt.data + 'Data')   # getattr()函数作用是生成一个类，调用相应.py文件的形式 1. 类名（.py文件的初始化参数） ————可实例化类，也可以直接调用类
    train_data = DataModel(opt.data_root, train=True)

    # 每条数据集为一个包，包内可能有一条句子可能有多条句子（实例），可能存在单标签或者多标签
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=collate_fn)

    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
    # optimizer = optim.Adadelta(model.parameters(), rho=1.0, eps=1e-6, weight_decay=opt.weight_decay)
    # train
    print("start training...")
    max_pre = -1.0
    max_rec = -1.0
    for epoch in range(opt.num_epochs):

        total_loss = 0
        for idx, (data, label_set) in enumerate(train_data_loader):
            label = [l[0] for l in label_set]

            if opt.use_gpu:
                label = torch.LongTensor(label).cuda()
            else:
                label = torch.LongTensor(label)

            data = select_instance(model, data, label)
            model.batch_size = opt.batch_size

            optimizer.zero_grad()

            out = model(data, train=True)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch < -1:
            continue
        true_y, pred_y, pred_p = predict(model, test_data_loader)
        all_pre, all_rec, fp_res = eval_metric(true_y, pred_y, pred_p)

        last_pre, last_rec = all_pre[-1], all_rec[-1]
        # if last_pre > 0.24 and last_rec > 0.24:
        save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, fp_res, opt=opt.print_opt)
        print('{} Epoch {} save pr'.format(now(), epoch + 1))
        if last_pre > max_pre and last_rec > max_rec:
            print("save model")
            max_pre = last_pre
            max_rec = last_rec
            model.save(opt.print_opt)

        print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, last_pre, last_rec))


def select_instance(model, batch_data, labels):

    model.eval()
    select_ent = []
    select_num = []
    select_sen = []
    select_pf = []
    select_pool = []
    select_mask = []
    for idx, bag in enumerate(batch_data):
        insNum = bag[1]
        label = labels[idx]
        max_ins_id = 0
        if insNum > 1:
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)

            out = model(data)

            #  max_ins_id = torch.max(torch.max(out, 1)[0], 0)[1]

            # PCNN模型基于一个假设，在一个bag中，至少有一个句子是被标注正确的。bag中每个实例经过全连接后，都会得到关于每个关系类别的概率。
            # 预测bag中所有实力表示目标关系的概率，再选取最高者作为包的表示。
            o=out[:, label]
            l=torch.max(o, 0)
            max_ins_id = l[1]

            if opt.use_gpu:
                #  max_ins_id = max_ins_id.data.cpu().numpy()[0]
                max_ins_id = max_ins_id.item()
            else:
                max_ins_id = max_ins_id.data.numpy()[0]

        max_sen = bag[2][max_ins_id]
        max_pf = bag[3][max_ins_id]
        max_pool = bag[4][max_ins_id]
        max_mask = bag[5][max_ins_id]

        select_ent.append(bag[0])
        select_num.append(bag[1])
        select_sen.append(max_sen)
        select_pf.append(max_pf)
        select_pool.append(max_pool)
        select_mask.append(max_mask)

    if opt.use_gpu:
        data = map(lambda x: torch.LongTensor(x).cuda(), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])
    else:
        data = map(lambda x: torch.LongTensor(x), [select_ent, select_num, select_sen, select_pf, select_pool, select_mask])

    model.train()
    return data


def predict(model, test_data_loader):

    model.eval()

    pred_y = []
    true_y = []
    pred_p = []
    for idx, (data, labels) in enumerate(test_data_loader):
        true_y.extend(labels)
        for bag in data:
            insNum = bag[1]
            model.batch_size = insNum
            if opt.use_gpu:
                data = map(lambda x: torch.LongTensor(x).cuda(), bag)
            else:
                data = map(lambda x: torch.LongTensor(x), bag)

            out = model(data)
            out = F.softmax(out, 1)
            max_ins_prob, max_ins_label = map(lambda x: x.data.cpu().numpy(), torch.max(out, 1))  # 返回最大可能的概率和下标
            tmp_prob = -1.0
            tmp_NA_prob = -1.0
            pred_label = 0
            pos_flag = False

            for i in range(insNum):
                if pos_flag and max_ins_label[i] < 1:
                    continue
                else:
                    if max_ins_label[i] > 0:
                        pos_flag = True
                        if max_ins_prob[i] > tmp_prob:  # 找出预测的一类标签的最大概率
                            pred_label = max_ins_label[i]
                            tmp_prob = max_ins_prob[i]
                    else:
                        if max_ins_prob[i] > tmp_NA_prob:
                            tmp_NA_prob = max_ins_prob[i]

            if pos_flag:
                pred_p.append(tmp_prob)
            else:
                pred_p.append(tmp_NA_prob)

            pred_y.append(pred_label)

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(true_y) == size
    # true_y, pred_y 真实的标签和预测的标签，pred_p 预测中的最终得到的最大概率
    model.train()
    return true_y, pred_y, pred_p


if __name__ == "__main__":
    # 生成命令行接口（Command Line Interfaces, CLIs）
    import fire
    fire.Fire(train)

