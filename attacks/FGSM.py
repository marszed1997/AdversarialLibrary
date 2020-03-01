import torch
from torch import nn

from model import Model
from config import args
from utils import model_op, eval_op
from datasets import dataset as dtst
from attacks.AttackSet import AttackBase


class FastGradientSignMethod(AttackBase):
    def __init__(self, net, dataset):
        super().__init__(net, dataset)

    @classmethod
    def AttackMethod(cls, net, data, label, **kwargs):

        if args.cuda:
            data = data.cuda()
            label = label.cuda()

        criterion = torch.nn.CrossEntropyLoss()

        net.eval()
        with torch.enable_grad():
            data.requires_grad = True
            output = net(data)
            loss = criterion(output, label)

            net.zero_grad()
            loss.backward()

            adv_data = data.detach() + (args.epsilon * torch.sign(data.grad.data)).detach()

        return torch.clamp(adv_data, 0., 1.)


if __name__ == "__main__":
    net = Model.resnet50()
    if args.cuda:
        net = net.cuda()
    model_op.load_model(net, "../" + args.std_CIFAR10Model_path)
    std_test_dtst = dtst.cifar10("../" + args.std_CIFAR10Dataset_dir).test_data()
    '''
    # generate adv_dataset on std_testset
    attack = FastGradientSignMethod(net, std_test_dtst)
    attack.save_adv_dataset(save_mode=True, save_dir="../" + args.FGSM_CIFAR10TestDataset_dir)
    '''
    '''
    # the adv accuracy of multipth_dataset
    mlpth_dtst = dtst.multipth_dataset("../" + args.FGSM_CIFAR10TestDataset_dir)
    _ = eval_op.get_acc(net, mlpth_dtst)
    print(_)
    '''
    '''
    # the adv accuracy of std_dataset
    _ = eval_op.get_adv_acc(net, std_test_dtst, FastGradientSignMethod.AttackMethod)
    print(_)
    '''
