import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Model
from config import args
from utils import model_op, eval_op
from defenses.Train import TrainBase
from datasets import dataset as dtst

from attacks.FGSM import FastGradientSignMethod as FGSM
from attacks.PGD import ProjectedGradientDescent as PGD


class AdvTrain(TrainBase):
    def __init__(self, net, dataset, model_path):
        super().__init__(net, dataset, model_path)

    def attack_method(cls, net, data, label, **kwargs):
        return FGSM.attack_method(net, data, label)

    def forward_prop(self, data, label):
        criterion = nn.CrossEntropyLoss()

        adv_data = self.attack_method(self.net, data, label)
        adv_data.requires_grad = False

        self.net.train()

        return 0.5 * criterion(self.net(data), label) + 1.5 * criterion(self.net(adv_data), label)


if __name__ == "__main__":
    net = Model.resnet50()
    train_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).train_data()
    test_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).test_data()
    TrainObject = AdvTrain(net, train_dataset, "../" + args.FGSM_CIFAR10Model_path)
    TrainObject.train(test_dataset, "adv")
