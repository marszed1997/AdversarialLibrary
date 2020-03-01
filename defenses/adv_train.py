import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Model
from config import args
from utils import model_op, eval_op
from datasets import dataset as dtst

from attacks.FGSM import FastGradientSignMethod as FGSM
from attacks.PGD import ProjectedGradientDescent as PGD


class adv_train:
    def __init__(self, dataset, net, adv_model_path):

        self.net = net
        if args.cuda:
            self.net = self.net.cuda()
        self.model_path = adv_model_path
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4)

        self.start_epoch, self.best_acc = 0, 0.
        if os.path.exists(self.model_path):
            self.start_epoch, self.best_acc = model_op.load_model(net, self.model_path, self.optimizer)
        else:
            print("no adv_model found")
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    def train(self, attack_method, test_dataset):

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.start_epoch, args.num_epochs):

            for param in self.optimizer.param_groups:
                param['lr'] = 0.02 * (0.2 ** (epoch // 25))

            total_num = 0.
            total_loss = 0.

            for data, label in tqdm(self.loader):
                adv_data = attack_method(self.net, data, label)

                self.net.train()

                if args.cuda:
                    data = data.cuda()
                    label = label.cuda()
                    adv_data = adv_data.cuda()

                output = self.net(data)
                adv_output = self.net(adv_data)
                loss = 0.5 * criterion(output, label) + 1.5 * criterion(adv_output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_num += label.size(0)
                total_loss += loss.item()

            cur_acc = eval_op.get_acc(self.net, test_dataset)
            cur_adv_acc = eval_op.get_adv_acc(self.net, test_dataset, attack_method)
            print("epoch: %d, loss: %.3f, acc: %.3f, adv_acc: %.3f" % (epoch, total_loss / total_num, cur_acc, cur_adv_acc))

            if cur_adv_acc > self.best_acc:
                self.best_acc = cur_adv_acc
                model_op.save_model(self.net, epoch + 1, self.best_acc, self.model_path, self.optimizer)


if __name__ == "__main__":
    net = Model.resnet50()
    train_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).train_data()
    test_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).test_data()
    Train = adv_train(train_dataset, net, "../" + args.FGSM_CIFAR10Model_path)
    Train.train(FGSM.AttackMethod, test_dataset)
    '''
    _ = eval_op.get_acc(net, test_dataset)
    print(_)
    '''
    '''
    mlpth_dtst = dtst.multipth_dataset("../" + args.FGSM_CIFAR10TestDataset_dir)
    _ = eval_op.get_acc(net, mlpth_dtst)
    print(_)
    '''
    '''
    __ = eval_op.get_adv_acc(net, test_dataset, PGD.AttackMethod)
    print(__)
    '''
