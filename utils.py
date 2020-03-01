import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader

from config import args


class model_op:

    @classmethod
    def save_model(cls, model, epoch, best_acc, save_path, optimizer=None):
        optim_dict = None
        if optimizer is not None:
            optim_dict = optimizer.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optim_dict
        }

        torch.save(state, save_path)
        print("model saved")

    @classmethod
    def load_model(cls, model, load_path, optimizer=None):
        if os.path.isfile(load_path):
            print("loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            model.load_state_dict(checkpoint['state_dict'])
            print("model acc: %.3f" % checkpoint['best_acc'])
            if optimizer is not None and 'optimizer' in checkpoint.keys():
                print("optimizer loaded")
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print("No optimizer found")
            return checkpoint['epoch'], checkpoint['best_acc']
        else:
            print("no saved model found")
            return 0, 0.


class eval_op:

    @classmethod
    def get_acc(cls, net, dataset):
        acc, num = 0., 0.
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        with torch.no_grad():
            net.eval()
            print("func: utils.eval_op.get_acc")
            for data, label in tqdm(loader):
                if args.cuda:
                    data = data.cuda()
                    label = label.cuda()
                output = net(data)
                # print(output)
                predicted = output.argmax(dim=1, keepdim=False)
                num += label.size(0)
                acc += (predicted.int() == label.int()).sum().item()

        return acc / num

    @staticmethod
    def get_adv_acc(net, dataset, attack_method):
        acc, num = 0., 0.
        loader = DataLoader(dataset, batch_size=10, shuffle=False)
        print("func: utils.eval_op.get_adv_acc")
        for data, label in tqdm(loader):
            with torch.enable_grad():
                adv_data = attack_method(net, data, label)
            with torch.no_grad():
                net.eval()
                if args.cuda:
                    label = label.cuda()
                output = net(adv_data)
                # print(output)
                predicted = output.argmax(dim=1, keepdim=False)
                num += label.size(0)
                acc += (predicted.int() == label.int()).sum().item()

        return acc / num


class image_op:

    @classmethod
    def draw_image(cls, data, mode, norm=False, BGR2RGB=False):  # shape=(size, size) or (3, size, size) mode="gray" or "RGB"
        if isinstance(data, torch.Tensor):
            img = data.numpy()
        else:
            assert isinstance(data, np.ndarray), "utils.draw_image type(array)"
            img = data
        if norm:
            img = np.clip(img, 0., 1.) * 255.

        if mode == "gray":
            plt.imshow(img, cmap="gray")
        else:
            assert mode == "RGB", "utils.draw_image mode"
            assert isinstance(BGR2RGB, bool), "utils.draw_image cvtColor"
            img = np.transpose(img, (1, 2, 0))
            if BGR2RGB:  # plt需要 RGB 的图片
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
        plt.axis('off')
        plt.show()

    @classmethod
    def load_image(cls, filename, mode):
        if mode == "gray":
            return cv2.imread(filename, flags=0)
        else:
            assert mode == "RGB", "utils.load_image mode"
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.reshape(1, img.shape[2], img.shape[0], img.shape[1])
            return img

    @classmethod
    def save_image(cls, data, filename, mode, norm=False, RGB2BGR=False):  # RGBtoBGR == 1: RGB->BGR
        if isinstance(data, torch.Tensor):
            img = data.numpy()
        else:
            assert isinstance(data, np.ndarray), "utils.save_image type(array)"
            img = data
        if norm:
            img = np.clip(img, 0., 1.) * 255.
        img = img.astype("uint8")
        if mode == "gray":
            cv2.imwrite("./images/cache/" + filename + ".jpg", img)
        else:
            assert mode == "RGB", "utils.save_image mode"
            assert RGB2BGR == 1 or RGB2BGR == 0, "utils.save_image cvtColor"
            img = np.transpose(img, (1, 2, 0))
            if RGB2BGR:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite("./images/cache/" + filename + ".jpg", img)    # imwrite 需要 BGR 图片

'''
class TensorX_op:
    def __init__(self, save_path):
        self.writer = SummaryWriter(logdir=save_path)

    def save_training(self, param_list, *kwargs):
        # self.param_list = param_list
        self.writer.add_hparams({param_list[index]:param for index, param in enumerate(kwargs)})

    def save_embedding(self, data, logit_data, label, epoch):
        data = data.cpu()
        logit_data = logit_data.cpu()
        label = label.cpu()
        self.writer.add_embedding(
            logit_data,
            metadata=[_ for _ in range(10)],
            label_img=data,
            global_step=epoch)
'''
