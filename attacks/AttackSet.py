import abc
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config import args


class AttackBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, net, dataset):
        self.net = net
        self.dataset = dataset
        self.loader = None
        if self.dataset is not None:
            self.loader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False)

    @classmethod
    @abc.abstractmethod
    def AttackMethod(cls, net, data, label, **kwargs):
        pass

    def save_adv_dataset(self, save_mode=False, save_dir=None):
        # data_list = [[] for _ in range(args.num_classes)]
        adv_data_list = [[] for _ in range(args.num_classes)]

        print("func: AttackSet.save_adv_dataset, generating adv_examples:")
        for data, label in tqdm(self.loader):

            adv_data = self.AttackMethod(self.net, data, label)
            adv_data = adv_data.cpu()

            for i in range(data.size(0)):
                # data_list[label[i].item()].append(data[i:i + 1])
                adv_data_list[label[i].item()].append(adv_data[i:i + 1])

        if save_mode is False:
            return

        for i in range(args.num_classes):
            # data = torch.cat(data_list[i], dim=0)
            adv_data = torch.cat(adv_data_list[i], dim=0)

            state = {
                "len": adv_data.size(0),
                # "data": data,
                "adv_data": adv_data,
                "label": i * torch.ones(adv_data.size(0))
            }

            torch.save(state, save_dir + str(i) + ".pth")
            print("save ori_data and adv_data of label: %d, len: %d" %(i, adv_data.size(0)))