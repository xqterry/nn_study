import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
if __name__ == '__main__':

    learning_rate = 1e-3
    batch_size = 100
    epoches = 50
    trans_img = transforms.Compose([
            transforms.ToTensor()
        ])
    trainset = MNIST('./data', train=True, transform=trans_img, download=True)
    testset = MNIST('./data', train=False, transform=trans_img, download=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    # build network
    class Lenet(nn.Module):
        def __init__(self):
            super(Lenet, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 6, 3, stride=1, padding=1),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5, stride=1, padding=0),
                nn.MaxPool2d(2, 2)
            )
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(84, 10)
            )
        def forward(self, x):
            out = self.conv(x)
            # print("conv 1 end")
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    lenet = Lenet()
    lenet.cuda()
    print("gpu #1 ok")
    criterian = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.SGD(lenet.parameters(), lr=learning_rate)
    # train
    for i in range(epoches):
        running_loss = 0.
        running_acc = 0.
        for (img, label) in trainloader:
            img = Variable(img).cuda()
            label = Variable(label).cuda()
            optimizer.zero_grad()
            output = lenet(img)
            loss = criterian(output, label)
            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            _, predict = torch.max(output, 1)
            correct_num = (predict == label).sum()
            running_acc += correct_num.data[0]
        running_loss /= len(trainset)
        running_acc /= len(trainset)
        print("[%d/%d] Loss: %.5f, Acc: %.2f" %(i+1, epoches, running_loss, 100*running_acc))

    lenet.eval()
    testloss = 0.
    testacc = 0.
    for (img, label) in testloader:
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        output = lenet(img)
        loss = criterian(output, label)
        testloss += loss.data[0]
        _, predict = torch.max(output, 1)
        num_correct = (predict == label).sum()
        testacc += num_correct.data[0]
    testloss /= len(testset)
    testacc /= len(testset)
    print("Test: Loss: %.5f, Acc: %.2f %%" % (testloss, 100 * testacc))