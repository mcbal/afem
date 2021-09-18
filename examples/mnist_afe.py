# Example of a VectorSpinAttention module trained using
# approximate free energy loss and then tested using inference
# on classification site starting from random vector (or zeros).

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
    for name, param in model.named_parameters():
        if 'attention' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for _, (data, target) in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {epoch}')
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            loss = model(data, target)

            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            # loss = F.cross_entropy(output, target)
            # preds = output.argmax(dim=1, keepdim=True)
            # correct = preds.eq(target.view_as(preds)).sum().item()
            # accuracy = correct / target.shape[0]
            loss.backward()

            # print(torch.linalg.norm(model.attention.spin_model._J))
            # breakpoint()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            tqdm_loader.set_postfix(afe_loss=f'{loss.item():.4f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # with torch.no_grad():
    for param in model.parameters():
        param.requires_grad = False
    counter = 0

    for data, target in test_loader:

        bsz = data.size(0)

        y = torch.randn(bsz, 1, 64).requires_grad_()
        optimizer = optim.Adam([y], lr=1e-3)

        for i in range(10):
            data, target = data.clone().to(device), target.clone().to(device)
            optimizer.zero_grad()

            loss, preds = model(data, y)
            # print(preds.indices.shape)

            loss.backward()
            optimizer.step()

            # preds = output.argmax(dim=1, keepdim=True)
        bla = preds.indices.eq(target.view_as(preds.indices)).sum().item()

        correct += bla
        counter += bsz
        print(i, loss, correct / counter)
        del optimizer

    test_loss /= len(test_loader.dataset)
    print(
        '\n✨ Test set: Average loss: {:.4f}, Accuracy: {}/{})\n'.format(
            0.0,
            correct,
            len(test_loader.dataset),
        )
    )


class MNISTNet(nn.Module):
    def __init__(self, dim=64, dim_conv=32, num_spins=16+1):
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
        # self.cls_token = nn.Parameter(torch.randn(1, 32, dim))
        # self.cls_emb = torch.randn(10, dim)
        self.cls_emb = nn.Embedding(10, dim)
        self.attention = VectorSpinAttention(
            num_spins=num_spins, dim=dim, pre_norm=True, beta=1.0, use_scalenorm=True, J_symmetric=True,
            J_traceless=True, J_add_external=False)
        self.final = nn.Linear(10, dim)
        self.t0 = 1.0

        # print(self.t0)
        # breakpoint()

        self.prev_J = None

    def forward(self, x, y):
        x = self.to_patch_embedding(x)
        # x = torch.cat((x, torch.zeros((x.shape[0], 32, x.shape[-1]))), dim=1)
        # cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        # x = torch.cat((x, cls_tokens), dim=1)

        # print(self.training)
        if self.training:
            # print(y)
            # y = torch.nn.functional.embedding(y, self.cls_emb).unsqueeze(1)
            y = torch.nn.functional.one_hot(y, num_classes=10).float()
            y = self.final(y).unsqueeze(1)
            # print(y)
            x = torch.cat((x, y), dim=1)

            afe, t_star = self.attention(x, t0=self.t0, return_magnetizations=False)

            # self.t0 = t_star[0][0].detach().clone() + 0.1

            # if self.prev_J is not None:
            # print(self.prev_J, torch.eig(self.prev_J).eigenvalues, torch.det(self.prev_J))
            # breakpoint()
            # self.prev_J = self.attention.spin_model.J(x)[0].detach().clone()

            return afe.mean() / x.size(1)  # because this returns
        else:
            # print(y.shape)
            x = torch.cat((x, y), dim=1)
            afe, _ = self.attention(x, t0=self.t0, return_magnetizations=False)
            # print(afe)
            dist = torch.norm(y.repeat(1, 10, 1) - self.cls_emb.weight.repeat(x.size(0), 1, 1), dim=-1)
            # print(dist.shape)
            preds = dist.topk(1, largest=False)
            return afe.mean() / x.size(1), preds


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

    train_optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # test(model, device, test_loader)
    for epoch in range(1, 30 + 1):
        train(model, device, train_loader, train_optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
