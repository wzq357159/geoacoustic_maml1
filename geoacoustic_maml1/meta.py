import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        :param config:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """
        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:  [task_num=1, setsz, frequency point]
        :param y_spt:  [b, setsz]
        :param x_qry:  [b, querysz, frequency point]
        :param y_qry:  [b, querysz]
        :return:
        """
        task_num, setsz, frepoint = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        losses_t = [0 for _ in range(self.update_step)]

        for i in range(task_num):

            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.mse_loss(logits.squeeze(1), y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            losses_t[0] += loss

            with torch.no_grad():
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.mse_loss(logits_q.squeeze(1), y_qry[i])
                losses_q[0] += loss_q

            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.mse_loss(logits_q.squeeze(1), y_qry[i])
                losses_q[1] += loss_q

            for k in range(1, self.update_step):
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.mse_loss(logits.squeeze(1), y_spt[i])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                losses_t[k] += loss

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.mse_loss(logits_q.squeeze(1), y_qry[i])
                losses_q[k + 1] += loss_q

        loss_q = losses_q[-1] / task_num

        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()
        out_q = list(map((lambda p: p.cpu().detach()), losses_q))
        mse_avg = np.array(out_q) / task_num



        out_t = torch.tensor(losses_t, device='cpu')
        mse_avg_t = np.array(out_t) / task_num

        return mse_avg, mse_avg_t

    def finetuning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:  [setsz, frequency point]
        :param y_spt:  [setsz]
        :param x_qry:  [querysz, frequency point]
        :param y_qry:  [querysz]
        :return:
        """
        querysz = x_qry.size(0)

        losses_q = [0 for _ in range(self.update_step_test + 1)]


        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.mse_loss(logits.squeeze(1), y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))


        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            loss_q = F.mse_loss(logits_q.squeeze(1), y_qry)
            losses_q[0] += loss_q

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            loss_q = F.mse_loss(logits_q.squeeze(1), y_qry)
            losses_q[1] += loss_q

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits.squeeze(1), y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))


            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q.squeeze(1), y_qry)
            losses_q[k + 1] += loss_q


        del net
        out_q = list(map((lambda p: p.cpu().detach()), losses_q))
        mse_avg = np.array(out_q)


        return mse_avg
