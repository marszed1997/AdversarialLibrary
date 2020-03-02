from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from config import args
from model import Model
from defenses.Train import TrainBase
from datasets import dataset as dtst
from attacks.PGD import ProjectedGradientDescent as PGD


class Trades(TrainBase):
    def __init__(self, net, dataset, model_path):
        super().__init__(net, dataset, model_path)

    def attack_method(cls, net, data, label, **kwargs):

        net.eval()
        criterion = nn.KLDivLoss()

        noise = 0.001 * torch.randn(data.size())
        if args.cuda:
            noise = noise.cuda()
        noise_data = data.detach() + noise.detach()
        with torch.enable_grad():
            for _ in range(10):
                noise_data.requires_grad = True
                loss = criterion(F.log_softmax(net(noise_data), dim=1),
                                 F.softmax(net(data), dim=1))
                grad = torch.autograd.grad(loss, noise_data)[0]
                noise_data = noise_data.detach() + 0.003 * torch.sign(grad)
                noise_data = torch.min(torch.max(noise_data, data - args.epsilon), data + args.epsilon)
                noise_data = torch.clamp(noise_data, 0.0, 1.0)
        return noise_data

    def forward_prop(self, net, data, label):
        adv_data = self.attack_method(net, data, label)
        adv_data.requires_grad = False

        net.train()
        net.zero_grad()
        logits = net(adv_data)
        loss_natural = nn.CrossEntropyLoss()
        criterion = nn.KLDivLoss()
        loss_robust = (1.0 / args.batch_size) * criterion(F.log_softmax(net(adv_data), dim=1),
                                                          F.softmax(net(data), dim=1))
        loss = loss_natural(logits, label) + 1. * loss_robust
        return loss


if __name__ == "__main__":
    net = Model.resnet50()
    if args.cuda:
        net = net.cuda()
    train_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).train_data()
    test_dataset = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).test_data()
    TradesObject = Trades(net, train_dataset, args.PGD_CIFAR10Model_path)
    TradesObject.train(test_dataset, "adv")
