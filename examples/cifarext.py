# Example of a VectorSpinAttention module acting on a "spin system" made up of
# the feature vectors of a tiny convolutional neural network together with a
# ViT-style "classification token". Final prediction is obtained from a linear
# layer acting on the output of the classification token.

import matplotlib.pyplot as plt
import numpy as np
from afem.modules import ScaleNorm
import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from einops.layers.torch import Rearrange

from afem.attention import VectorSpinAttention

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
writer = SummaryWriter(f"runs/debug_{datetime.now()}")


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    with tqdm(train_loader, unit='it') as tqdm_loader:
        for idx, (data, target) in enumerate(tqdm_loader):
            tqdm_loader.set_description(f'Epoch {epoch}')
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            loss_logp, logits, loss_xent, t_star = model(data, target)
            loss = loss_xent + loss_logp
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            preds = logits.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            accuracy = correct / target.shape[0]
            loss.backward()

            # print(model.cls_token.grad)
            # breakpoint()

            optimizer.step()
            tqdm_loader.set_postfix(
                loss_logp=f'{loss_logp.item():.4f}',
                loss_xent=f'{loss_xent.item():.4f}',
                accuracy=f'{accuracy:.4f}'
            )

            writer.add_scalar('t_star_avg', t_star.mean(), (epoch-1)*len(train_loader)+idx)
            writer.add_scalar('accuracy', accuracy, (epoch-1)*len(train_loader)+idx)
            writer.add_scalar('total_loss', loss.item(), (epoch-1)*len(train_loader)+idx)
            writer.add_scalar('loss_afem', loss_logp.item(), (epoch-1)*len(train_loader)+idx)
            writer.add_scalar('loss_xe', loss_xent.item(), (epoch-1)*len(train_loader)+idx)

            # for name, tensor in model.state_dict().items():
            #     # print(name, tensor)
            #     writer.add_histogram(model.__class__.__name__ + '.' + name,
            #                          tensor.clone().detach().cpu().numpy(), (epoch-1)*len(train_loader)+idx)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        with tqdm(test_loader, unit='it') as tqdm_loader:
            for idx, (data, target) in enumerate(tqdm_loader):
                data, target = data.to(device), target.to(device)
                loss_logp, logits, loss_xent, _ = model(data, target)
                loss = loss_xent + loss_logp
                test_loss += loss.item()
                preds = logits.argmax(dim=1, keepdim=True)
                correct += preds.eq(target.view_as(preds)).sum().item()

    test_loss /= len(test_loader)
    print(
        '\n✨ Test set: Average loss: {:.4f}, Accuracy: {}/{})\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
        )
    )


class CIFAR10Net(nn.Module):
    def __init__(self, dim=32, dim_conv=32, num_spins=16+16+1):
        super(CIFAR10Net, self).__init__()

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # -> dim_conv x 16 x 16
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # -> dim_conv x 8 x 8
            nn.Conv2d(dim_conv, dim_conv, kernel_size=3, padding=1),  # -> dim_conv x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # -> dim_conv x 4 x 4
            Rearrange('b c h w -> b (h w) c'),  # -> 16 x dim_conv
            nn.Linear(dim_conv, dim),  # -> 16 x dim
        )

        t0 = 0.75*torch.ones(num_spins)
        self.register_buffer('t0', t0)
        # self.cls_token = torch.randn(1, 1, dim)  # torch.zeros(1, 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 16+1, dim))
        self.attention = VectorSpinAttention(
            num_spins=num_spins,
            dim=dim,
            post_norm=True,
            use_scalenorm=True,
            J_add_external=True,
            J_traceless=True,
            J_symmetric=True,
        )
        self.final = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 10)
        )

    def forward(self, x, y=None):

        # # print labels
        # print(' '.join('%5s' % classes[y[j]] for j in range(4)))
        # # show images
        # imshow(torchvision.utils.make_grid(x[:4]))

        # breakpoint()

        x = self.to_patch_embedding(x)
        # print(x.shape)

        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((x, cls_tokens), dim=1)
        # x = torch.cat((x, torch.zeros((x.shape[0], 1, x.shape[-1]))), dim=1)

        out = self.attention(x, t0=self.t0, return_log_prob=True)
        x = out.magnetizations
        # print(torch.linalg.norm(x.mean(dim=1), dim=-1))
        # print(torch.linalg.norm(ScaleNorm(np.sqrt(32))(x.mean(dim=1)), dim=-1))

        logits = self.final(x[:, -1, :])
        # logits = self.final(out.magnetizations.mean(dim=1))

        # ret = self.final(x.mean(dim=1))

        loss_logp = -out.log_prob.mean() / self.attention.spin_model.num_spins
        # loss_jem = torch.log_softmax(logits, dim=-1)

        loss_xent = F.cross_entropy(logits, y) if y is not None else None

        return loss_logp, logits, loss_xent, out.t_star


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CIFAR10Net()
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

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = datasets.CIFAR10('.', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=args.bsz, num_workers=1, shuffle=True)
        test_data = datasets.CIFAR10('.', train=False, transform=transform)
        test_loader = DataLoader(test_data, batch_size=args.bsz, num_workers=1, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            if epoch % args.eval_every == 0:
                test(model, device, test_loader)
            if epoch % args.ckpt_every == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch:03}.pt'))
    elif args.mode == 'test':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        test_data = datasets.CIFAR10('.', train=False, transform=transform)
        test_loader = DataLoader(test_data, batch_size=args.bsz, num_workers=1, shuffle=False)
        test(model, device, test_loader)
    else:
        raise ValueError('How did you even get here? Run script with `train` or `test`.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser_train.add_argument("--bsz", type=int, default=50, help='Train batch size')
    parser_train.add_argument("--epochs", type=int, default=30, help='Number of epochs')
    parser_train.add_argument(
        "--save_dir", type=str, default=f'./mnist_vit_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}',
        help="Directory to store model checkpoints")
    parser_train.add_argument("--ckpt_every", type=int, default=1, help="Epochs between model saving")
    parser_train.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('model_path', type=str, help='File path to file containing model state dict')
    parser_test.add_argument("--bsz", type=int, default=50, help='Test batch size')

    args = parser.parse_args()
    main(args)
