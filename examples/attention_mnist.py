# Example of a VectorSpinAttention module acting on a "spin system" made up of
# the feature vectors of a tiny convolutional neural network together with a
# ViT-style "classification token". Final prediction is obtained from a linear
# layer acting on the response of the classification token.

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from einops.layers.torch import Rearrange

from afem.attention import VectorSpinAttention


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for _, (data, target) in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {epoch}')
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            preds = output.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            accuracy = correct / target.shape[0]
            loss.backward()
            optimizer.step()
            tqdm_loader.set_postfix(loss=f'{loss.item():.4f}', accuracy=f'{accuracy:.4f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            preds = output.argmax(dim=1, keepdim=True)
            correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        '\n✨ Test set: Average loss: {:.4f}, Accuracy: {}/{})\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
        )
    )


class MNISTNet(nn.Module):
    def __init__(self, dim=96, dim_conv=32, num_spins=16+1):
        super(MNISTNet, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, dim_conv, kernel_size=3),  # -> dim_conv x 26 x 26
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # -> dim_conv x 12 x 12
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3),  # -> dim_conv x 10 x 10
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),  # -> dim_conv x 4 x 4
            Rearrange('b c h w -> b (h w) c'),  # -> (4 x 4) x dim_conv
            nn.Linear(dim_conv, dim),  # -> (4 x 4) x dim
        )
        self.attention = VectorSpinAttention(num_spins=num_spins, dim=dim, pre_norm=True, post_norm=True)
        self.final = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = torch.cat((x, torch.zeros((x.shape[0], 1, x.shape[-1]))), dim=1)
        x = self.attention(x)[0]
        return self.final(x[:, -1, :])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_data = datasets.MNIST(
        '.', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('.', train=False, transform=transform)
    train_loader = DataLoader(train_data,
                              batch_size=60,
                              num_workers=1,
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             batch_size=60,
                             num_workers=1,
                             shuffle=False)

    model = MNISTNet().to(device)

    print(
        f'\n✨ Initialized {model.__class__.__name__} ({sum(p.nelement() for p in model.parameters())} params) on {device}.'
    )  # number of model parameters may be an overestimate if weights are made symmetric/traceless inside model

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, 30 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
