import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class Residual(nn.Module):

    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk


def accuracy(y_hat, y):
    acc = (y_hat.argmax(dim=1)==y).type(torch.float32).sum().item()
    return acc


def validation(net, dataloader, loss):
    num_batches = len(dataloader)
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            test_acc += accuracy(y_hat, y)
            test_loss += loss(y_hat, y)
    test_acc /= len(dataloader.dataset)
    test_loss /= num_batches
    return test_acc, test_loss


def train(net, train_dataloader, test_dataloader, lr, num_epochs):
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    print('Start trainging ...')
    print('Training on', device)

    for epoch in range(num_epochs):
        print('epoch', epoch+1)
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            trainer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            if batch%200 == 0:
                train_loss = l.item()
                current_batch = batch * len(X)
                train_acc = accuracy(y_hat, y) / len(X)
                print('Train loss: %.4f\tTrain acc:%.4f\t[%d/%d]' % (train_loss, train_acc, current_batch, len(train_dataloader.dataset)))
        with torch.no_grad():
            test_acc, test_loss = validation(net, test_dataloader, loss)
        print('Test loss: %.4f\tTest acc: %.4f' % (test_loss, test_acc))
    print('Done!')


# 数据集
trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(224)
])

train_data = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=False
)

test_data = torchvision.datasets.FashionMNIST(
    root='../data', train=False, transform=trans, download=False
)

batch_size = 64

train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size
)

test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size
)


# ResNet
block1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
block2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
block3 = nn.Sequential(*resnet_block(64, 128, 2))
block4 = nn.Sequential(*resnet_block(128, 256, 2))
block5 = nn.Sequential(*resnet_block(256,512, 2))
net = nn.Sequential(
    block1, block2, block3, block4, block5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

# 训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
train(net, train_dataloader, test_dataloader, lr=1e-2, num_epochs=5)