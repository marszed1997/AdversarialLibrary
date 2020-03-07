import torch
from torch import nn, optim
import torch.nn.functional as F

from config import args
from model import Model
from utils import model_op, eval_op
from defenses.Train import TrainBase
from datasets import dataset as dtst


class StdTrain(TrainBase):
    def __init__(self, net, dataset, model_path):
        super().__init__(net, dataset, model_path)

    def attack_method(cls, net, data, label, **kwargs):
        pass

    def forward_prop(self, data, label):
        self.net.train()
        self.net.zero_grad()

        return nn.CrossEntropyLoss()(self.net(data), label)


if __name__ == "__main__":
    net = Model.resnet50()
    train_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).train_data()
    test_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).test_data()
    TrainObject = StdTrain(net, train_dataset, "../" + args.std_CIFAR10Model_path)
    # TradesObject.train(test_dataset, "std")
    acc = TrainObject.print_acc(net, test_dataset)

