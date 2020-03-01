import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from config import args
from utils import image_op

class cifar10:
    def __init__(self, save_path):
        self.save_path = save_path

        self.transforms_train = transforms.Compose([
            # transforms.RandomCrop(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomResizedCrop(size=(32, 32), scale=(0.6, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        self.transforms_test = transforms.Compose([
            transforms.ToTensor()
        ])

    def train_data(self):
        return datasets.CIFAR10(root=self.save_path, train=True, transform=self.transforms_train, download=False)

    def test_data(self):
        return datasets.CIFAR10(root=self.save_path, train=False, transform=self.transforms_test, download=False)


class multipth_dataset(Dataset):
    def __init__(self, multi_pth_dir):
        self.start = 0
        self.index_list = []
        # self.data = [[] for i in range(args.num_classes)]
        self.adv_data = [[] for i in range(args.num_classes)]
        self.label = [[] for i in range(args.num_classes)]
        for i in range(args.num_classes):
            pth_dict = torch.load(multi_pth_dir + str(i) + '.pth')
            # self.data[i] = pth_dict['data']
            self.adv_data[i] = pth_dict['adv_data']
            self.label[i] = pth_dict['label']
            self.index_list.append((self.start, self.start + pth_dict['adv_data'].size(0)))
            self.start = self.start + pth_dict['adv_data'].size(0)

    def __getitem__(self, index):
        for i in range(args.num_classes):
            start, end = self.index_list[i]
            if start <= index < end:
                id = index - start
                return self.adv_data[i][id], self.label[i][id]

    def __len__(self):
        return self.start


if __name__ == "__main__":
    '''
    # check if adv_example is ok
    pth_dataset = multipth_dataset("cifar10_PGD/")
    pth_loader = DataLoader(pth_dataset, batch_size=10, shuffle=True)
    for data, label in pth_loader:
        data = data
        print(label[0].item())
        image_op.draw_image(data[0], "RGB")
        break
    '''