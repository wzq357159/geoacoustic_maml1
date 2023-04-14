import torch, os
import numpy as np
from geodata import geodata
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse

from meta import Meta


def mean_confidence_interval(mse_avg, confidence=0.95):
    n = mse_avg.shape[0]
    m, se = np.mean(mse_avg), scipy.stats.sem(mse_avg)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    np.set_printoptions(threshold=np.inf)
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('linear', [256, 25]),
        ('relu', [True]),
        ('linear', [1024, 256]),
        ('relu', [True]),
        ('dropout', []),
        ('linear', [256, 1024]),
        ('relu', [True]),
        ('linear', [25, 256]),
        ('relu', [True]),
        ('linear', [1, 25])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    testdata = np.zeros((12, args.update_step_test + 1))
    supportdata = np.zeros((40, args.update_step))
    querydata = np.zeros((40, args.update_step + 1))

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    geo = geodata(path='./finalgeoall1.csv', batchsz=400, setsz=350, querysz=50, param='c_s')
    geo_test = geodata(path='./finalgeoall1.csv', batchsz=20, setsz=350, querysz=50, param='c_s')

    db_test_ini = DataLoader(geo_test, 1, shuffle=True, num_workers=0, pin_memory=True)
    mse_all_test = []
    for x_spt, y_spt, x_qry, y_qry in db_test_ini:
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        mse_avg = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
        mse_all_test.append(mse_avg)
    mse_avg = np.array(mse_all_test).mean(axis=0).astype(np.float32)
    testdata[0] = mse_avg
    print('Initial Test mse:', mse_avg)

    supportdata_num = 0
    testdata_num = 1

    db = DataLoader(geo, args.task_num, shuffle=True, num_workers=0, pin_memory=True)
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
        mse_avg, mse_avg_t = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 10 == 0:
            supportdata[supportdata_num] = mse_avg_t
            querydata[supportdata_num] = mse_avg
            supportdata_num += 1
            print('step:', step, '\ttraining support mse:', mse_avg_t)
            print('step:', step, '\ttraining mse:', mse_avg)

        if step % 39 == 0:  # evaluation
            db_test = DataLoader(geo_test, 1, shuffle=True, num_workers=0, pin_memory=True)
            mse_all_test = []
            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                    x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
                mse_avg = maml.finetuning(x_spt, y_spt, x_qry, y_qry)
                mse_all_test.append(mse_avg)
            mse_avg = np.array(mse_all_test).mean(axis=0).astype(np.float32)

            testdata[testdata_num] = mse_avg
            testdata_num += 1

            print('Test mse:', mse_avg)
    np.savetxt('supportdata.csv', supportdata, delimiter=',')
    np.savetxt('querydata.csv', querydata, delimiter=',')
    np.savetxt('testdata.csv', testdata, delimiter=',')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--batchsz', type=int, help='batchsz', default=400)
    argparser.add_argument('--setsz', type=int, help='setsz', default=350)
    argparser.add_argument('--querysz', type=int, help='querysz', default=50)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1000)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=1000)

    args = argparser.parse_args()

    main()
