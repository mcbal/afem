# Example of a VectorSpinAttention module trained using
# approximate free energy loss and then tested using inference
# on classification site starting from random vector (or zeros).

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from einops.layers.torch import Rearrange

from afem.attention import VectorSpinAttention


def sample(model, device):
    x = ((torch.zeros(1, 1, 28, 28)-0.1307)/0.3081).requires_grad_()
    # x = torch.rand(1, 1, 28, 28).requires_grad_()
    # print(x)
    eps = 0.01
    langevin_noise = torch.distributions.Normal(
        torch.zeros(x.shape),
        torch.ones(x.shape) * eps
    )
    for name, param in model.named_parameters():
        param.requires_grad = False

    for i in range(300):
        fun = model(x)[0]
        print(i, fun)
        grad_ea = torch.autograd.grad(fun, x)[0]
        # print(grad_ea)
        # grad_ea = torch.clamp(grad_ea, -self.opt.grad_clip_sampling, self.opt.grad_clip_sampling)
        x = x - 1.0 / 2.0 * grad_ea + langevin_noise.sample()
        x = x.clamp(-0.1307/0.3081, (1.0-0.1307)/0.3081)
        # if i % 10 == 0:
        #     image = ((0.1307+x*0.3081).clone().detach()).squeeze(0).squeeze(0).numpy()
        #     # if i % 20 == 0:
        #     plt.imshow(image, cmap='gray')
        #     plt.colorbar()
        #     plt.show()

        # x = torch.clamp(x, -0.1307/0.3081, (1.0-0.1307)/0.3081)
    image = ((0.1307+x*0.3081).clone().detach()).squeeze(0).squeeze(0).numpy()
    # if i % 20 == 0:
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()
    for name, param in model.named_parameters():
        param.requires_grad = True


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    # for name, param in model.named_parameters():
    #     param.requires_grad = True

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for idx, (data, target) in enumerate(tqdm_loader):

            # plt.imshow((0.1307+0.3081*data[0].clone().detach().squeeze(0).squeeze(0)).numpy(), cmap='gray')
            # plt.colorbar()
            # plt.show()

            tqdm_loader.set_description(f'Epoch {epoch}')
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            loss_afem, logits, loss_xe = model(data, target)

            loss = loss_afem + loss_xe
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            # loss = F.cross_entropy(output, target)
            preds = logits.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            accuracy = correct / target.shape[0]
            loss.backward()

            # print(model.attention.spin_model._J.grad)
            # breakpoint()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            # print(model.attention.spin_model._J.grad)

            optimizer.step()
            tqdm_loader.set_postfix(loss_afem=f'{loss_afem.item():.4f}',
                                    loss_xe=f'{loss_xe.item():.4f}', accuracy=f'{accuracy:.4f}')

            if idx % 1000 == 0:
                sample(model, device)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            lossa, logits, lossb = model(data, target)
            loss = lossa + lossb
            test_loss += loss.item()
            preds = logits.argmax(dim=1, keepdim=True)
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
    def __init__(self, dim=32, dim_conv=24, num_spins=16+1):
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.cls_emb = torch.randn(10, dim)
        # self.cls_emb = nn.Embedding(10, dim)
        self.attention = VectorSpinAttention(
            num_spins=num_spins, dim=dim, pre_norm=True, post_norm=True, beta=1.0, use_scalenorm=True,
            J_symmetric=True, J_traceless=True, J_add_external=True)
        self.final = nn.Linear(dim, 10)
        self.t0 = 1.0

        # print(self.t0)
        from afem.modules import ScaleNorm
        self.prenorm = ScaleNorm(dim)
        # breakpoint()

        self.prev_J = None

    def forward(self, x, y=None):
        # print(x[0])
        # breakpoint()
        # print(x[0])
        # plt.imshow((0.1307+0.3081*x[0].clone().detach().squeeze(0).squeeze(0)).numpy(), cmap='gray')
        # plt.colorbar()
        # plt.show()

        x = self.to_patch_embedding(x)
        # x = torch.cat((x, torch.zeros((x.shape[0], 32, x.shape[-1]))), dim=1)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_tokens), dim=1)
        # x = torch.cat((x, torch.zeros((x.shape[0], 1, x.shape[-1]))), dim=1)
        # print(self.training)
        # if self.training:
        # print(y)
        # y = torch.nn.functional.embedding(y, self.cls_emb).unsqueeze(1)
        # y = self.cls_emb(y).unsqueeze(1)
        # print(y)
        # x = torch.cat((x, y), dim=1)

        resps, afe, t_star = self.attention(x, t0=self.t0, return_magnetizations=True)
        # print(t_star)
        # if (t_star > 1.4).any():
        #     self.t0 = 2.0

        logits = self.final(resps[:, -1, :])

        loss_afem = self.attention.spin_model.loss(t_star, self.prenorm(x)).mean()  # / (33)
        # loss_afem = torch.randn(1, 1)
        out = (loss_afem, logits)

        if y is not None:
            loss_xe = F.cross_entropy(logits, y)
            out += (loss_xe,)

        return out
        # else:
        #     # print(y.shape)
        #     x = torch.cat((x, y), dim=1)
        #     afe, t_star_ = self.attention(x, t0=self.t0, return_magnetizations=False)
        #     # print(afe)
        #     dist = torch.norm(y.repeat(1, 10, 1) - self.cls_emb.weight.repeat(x.size(0), 1, 1), dim=-1)
        #     # print(dist.shape)
        #     preds = dist.topk(1, largest=False)
        #     return self.attention.spin_model.loss(t_star_, x).mean(), preds


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

    train_optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # test(model, device, test_loader)
    for epoch in range(1, 30 + 1):

        # sample(model, device)

        train(model, device, train_loader, train_optimizer, epoch)

        test(model, device, test_loader)


if __name__ == '__main__':
    main()
