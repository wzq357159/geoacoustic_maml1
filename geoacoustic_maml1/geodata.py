import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import random


class geodata(Dataset):
    def __init__(self, path,  batchsz, setsz, querysz, param, startindx=0):
        self.batchsz = batchsz  # batch of set,
        self.setsz = setsz      # num of samples per set for training
        self.querysz = querysz   # number of samples per set for evaluation
        self.startidx = startindx  # index label not from 0, but from startidx
        self.param = param  # c_s or alpha

        csvdata = self.loadCSV(path)

        self.data = []
        self.pressure2idx = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [(pressure list),...]
            self.pressure2idx[k] = i + self.startidx  # {(h_s,...):0,(h_s,...):1}

        self.task2idx = {}   # {(h_s,h_sr,c_b):[0,1,2,...]} each is a task
        for i, (k, v) in enumerate(self.pressure2idx.items()):
            tasklabel = k[0:2]
            if tasklabel in self.task2idx.keys():
                self.task2idx[tasklabel].append(v)
            else:
                self.task2idx[tasklabel] = [v]

        self.create_batch(self.batchsz)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param csvf: csv file path
        :return: {(h_s,h_sr,c_b,c_s,rou,a):(pressure list)}
        """
        dictdata = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(csvreader):
                label = tuple(map(float, row[0:4]))
                pressure = tuple(map(float, row[4:29]))
                dictdata[label] = pressure
        return dictdata

    def create_batch(self, batchsz):
        """
        create batch for meta-learning
        :param batchsz:
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(batchsz):
            selected_task = random.sample(self.task2idx.keys(), 1)
            selected_idx = self.task2idx[selected_task[0]]
            np.random.shuffle(selected_idx)
            indexDtest = selected_idx[:self.querysz]
            indexDtrain = selected_idx[self.querysz:]
            random.shuffle(indexDtrain)
            random.shuffle(indexDtest)

            self.support_x_batch.append(indexDtrain)
            self.query_x_batch.append(indexDtest)

    def __getitem__(self, index):
        """
        There is only one task(set) in the batch(index = 0 always), and the function returns the input and output of the task
        :param index:
        :return:
        """

        support_x = torch.FloatTensor(self.setsz, 25)   # 50 frequency points
        support_y = torch.FloatTensor(self.setsz)
        query_x = torch.FloatTensor(self.querysz, 25)
        query_y = torch.FloatTensor(self.querysz)
        for i, e in enumerate(self.support_x_batch[index]):
            support_x[i] = torch.from_numpy(np.array(self.data[e]))

        for i, e in enumerate(self.query_x_batch[index]):
            query_x[i] = torch.from_numpy(np.array(self.data[e]))

        if self.param == 'c_s':
            for i, e in enumerate(self.support_x_batch[index]):
                support_y[i] = list(self.pressure2idx.keys())[list(self.pressure2idx.values()).index(e)][2]
            for i, e in enumerate(self.query_x_batch[index]):
                query_y[i] = list(self.pressure2idx.keys())[list(self.pressure2idx.values()).index(e)][2]
        else:
            for i, e in enumerate(self.support_x_batch[index]):
                support_y[i] = list(self.pressure2idx.keys())[list(self.pressure2idx.values()).index(e)][3]
            for i, e in enumerate(self.query_x_batch[index]):
                query_y[i] = list(self.pressure2idx.keys())[list(self.pressure2idx.values()).index(e)][3]

        return support_x, support_y, query_x, query_y

    def __len__(self):
        return self.batchsz










