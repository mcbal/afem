# Example of a VectorSpinAttention module acting on a "spin system" made up of
# the feature vectors of a tiny convolutional neural network together with a
# ViT-style "classification token". Final prediction is obtained from a linear
# layer acting on the output of the classification token.

import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from einops.layers.torch import Rearrange

from afem.attention import VectorSpinAttention
from afem.utils import exists


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for _, (data, target) in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {epoch}')
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
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
        with tqdm(test_loader, unit='it') as tqdm_loader:
            for idx, (data, target) in enumerate(tqdm_loader):
                tqdm_loader.set_description(f'Testing batch {idx} / {len(test_loader)}')
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
    def __init__(self, dim=32, dim_conv=32, num_spins=9+1):
        super(MNISTNet, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 28 x 28
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # -> dim_conv x 14 x 14
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 14 x 14
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # -> dim_conv x 7 x 7
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 7 x 7
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),  # -> dim_conv x 3 x 3
            Rearrange('b c h w -> b (h w) c'),  # -> 9 x dim_conv
            nn.Linear(dim_conv, dim),  # -> 9 x dim
        )

        self.t0 = 0.75*torch.ones(num_spins)
        self.attention = VectorSpinAttention(
            num_spins=num_spins,
            dim=dim,
            post_norm=True,
            J_add_external=True,
            J_traceless=False,
            J_symmetric=False
        )
        self.final = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = torch.cat((x, torch.zeros((x.size(0), 1, x.size(-1)))), dim=1)
        x = self.attention(x, t0=self.t0).magnetizations
        return self.final(x[:, -1, :])


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor()])  # MNIST data range is [0, 1]
    test_data = datasets.MNIST('.', train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.bsz, num_workers=1, shuffle=False)

    model = MNISTNet()
    if hasattr(args, 'model_path'):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(
            f'\n✨ Loaded model weights for {model.__class__.__name__} ({sum(p.nelement() for p in model.parameters())} params) on {device}. '
            '(Number of model parameters may be an overestimate if weights are made symmetric/traceless inside model.)\n')
    else:
        print(
            f'\n✨ Initialized {model.__class__.__name__} ({sum(p.nelement() for p in model.parameters())} params) on {device}. '
            '(Number of model parameters may be an overestimate if weights are made symmetric/traceless inside model.)\n')
    model.to(device)

    if args.mode == 'train':
        os.makedirs(args.save_dir, exist_ok=True)
        train_data = datasets.MNIST('.', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=args.bsz, num_workers=1, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            if epoch % args.eval_every == 0:
                test(model, device, test_loader)
            if epoch % args.ckpt_every == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch:03}.pt'))
    else:
        test(model, device, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser_train.add_argument("--bsz", type=int, default=60, help='Train batch size')
    parser_train.add_argument("--epochs", type=int, default=30, help='Number of epochs')
    parser_train.add_argument(
        "--save_dir", type=str, default=f'./mnist_vit_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}',
        help="Directory to store model checkpoints")
    parser_train.add_argument("--ckpt_every", type=int, default=1, help="Epochs between model saving")
    parser_train.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path', type=str, help='File path to file containing model state dict')
    parser_test.add_argument("--bsz", type=int, default=60, help='Test batch size')

    args = parser.parse_args()
    main(args)
