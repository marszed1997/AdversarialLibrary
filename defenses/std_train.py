import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Model
from config import args
from datasets import dataset as dl
from utils import model_op, eval_op


def std_train(net):
    train_data = dl.cifar10("../" + args.CIFAR10_path).train_data()
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    test_data = dl.cifar10("../" + args.CIFAR10_path).test_data()
    # train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    start_epoch, best_acc = model_op.load_model(net, "../" + args.std_model_path)

    for epoch in range(start_epoch, args.num_epochs):

        optimizer = optim.SGD(net.parameters(), lr=args.learning_rate * (0.1 ** (epoch // 20)), weight_decay=5e-4)

        total_num = 0.
        total_loss = 0.
        for data, label in train_loader:
            if args.cuda:
                data = data.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            output = net(data)

            # output = nn.Softmax(dim=1)(output)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            total_num += label.size(0)
            total_loss += loss.item()

        cur_acc = eval_op.get_acc(net, test_data)
        print("epoch: %d, loss: %.3f, acc: %.3f" % (epoch, total_loss/total_num, cur_acc))

        if cur_acc > best_acc:
            best_acc = cur_acc
            model_op.save_model(net, epoch + 1, best_acc, "../" + args.std_model_path)


if __name__ == "__main__":
    '''
    net = model.resnet50()
    if args.cuda:
        net = net.cuda()
    std_train(net)
    '''
    net = Model.resnet50()
    if args.cuda:
        net = net.cuda()
    std_train(net)
    # model_op.load_model(net, "../" + args.std_model_path)
