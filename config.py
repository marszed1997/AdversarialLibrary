import argparse

import torch


parser = argparse.ArgumentParser(description="config")

parser.add_argument('--std_CIFAR10Dataset_dir', default='datasets/cifar10/')

parser.add_argument('--FGSM_CIFAR10TestDataset_dir', default='datasets/cifar10_FGSM/')

parser.add_argument('--PGD_CIFAR10TestDataset_dir', default='datasets/cifar10_PGD/')

parser.add_argument('--cuda', default=torch.cuda.is_available())

parser.add_argument('--num_epochs', type=int, default=300)

parser.add_argument('--learning_rate', type=float, default=0.02)

parser.add_argument('--std_CIFAR10Model_path', default="model/cifar10.pth")

parser.add_argument('--FGSM_CIFAR10Model_path', default="model/cifar10_FGSM.pth")

parser.add_argument('--PGD_CIFAR10Model_path', default="model/cifar10_PGD.pth")

parser.add_argument('--epsilon', type=float, default=0.02)

parser.add_argument('--batch_size', type=int, default=30)

parser.add_argument('--num_classes', type=int, default=10)

args = parser.parse_args()
