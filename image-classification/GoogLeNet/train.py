import torch
import torchvision
from torch import nn
from model import GoogLeNet



def train(dataloader, net, loss, optim):
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optim.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.backward()
        optim.step()
        if batch % 100 == 0:
            running_loss = l.item()
            current_batch = batch * len(X)
            print('Train loss: %.4f, [%d/%d]' % (running_loss, current_batch, len(dataloader.dataset)))


def test(dataloader, net, loss):
    num_batches = len(dataloader)
    val_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            val_loss += loss(y_hat, y).item()
            acc += (y_hat.argmax(dim=1)==y).type(torch.float32).sum().item()
    val_loss /= num_batches
    acc /= len(dataloader.dataset)
    print('Test accuracy: %.4f, Test average loss: %.4f' % (acc, val_loss))


def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)




trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(224)
])
train_data = torchvision.datasets.FashionMNIST(
    root='../data', train=True,
    download=False, transform=trans
)
test_data = torchvision.datasets.FashionMNIST(
    root='../data', train=False,
    download=False, transform=trans
)
print('The number of training data:', len(train_data))
print('The number of test data:', len(test_data))


batch_size = 64
train_data_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size
)
test_data_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size
)
for X, y in test_data_dataloader:
    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)
    break


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = GoogLeNet().apply(init_weights).to(device)
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)


num_epochs = 5
print('Starting training ...')
print('Training on', device)
for epoch in range(num_epochs):
    print('epoch %d' % (epoch+1))
    train(train_data_dataloader, net, loss, optim)
    test(test_data_dataloader, net, loss)
print('Done!')