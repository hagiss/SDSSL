import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
import torch.nn.functional as F
from einops import repeat
from torchmetrics import Accuracy

from torchvision import transforms as T
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
import random
import torch.distributed as dist
from torchvision import models as torchvision_models

import utils
import json
import math
import numpy as np
import vision_transformer as vits
from byol_pytorch import NetWrapper
import fine_tune
import sys
from pl_train_moco import PLLearner
from tqdm import tqdm

from PIL import Image


def default(val, def_val):
    return def_val if val is None else val


def count_parameters(m):
    return sum(p.numel() for p in m.parameters())


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Tuner(pl.LightningModule):
    def __init__(self, model, embed_dim, total_batch_size, length, lr=0.001):
        super().__init__()

        self.model = model
        # dim_mlp = self.model.fc[0].in_features
        # self.model.fc = nn.Identity()
        # self.model.train()
        self.fc = nn.ModuleList([nn.Linear(embed_dim, 1000) for _ in range(12)])

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.optim = torch.optim.SGD(
            self.fc.parameters(),
            lr * total_batch_size / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        self.scheduler = utils.cosine_scheduler(
            lr * total_batch_size / 256.,
            0,
            100, length
        )

        # self.optim = LARS(self.optim, eps=0.0)

        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=length, eta_min=0, last_epoch=-1)
        # w = scheduler.LinearWarmup(self.optim, warmup_steps=args.warmup, last_epoch=-1)
        # sched = scheduler.Scheduler(sched, w)
        # sched.optimizer = self.optim
        # self.scheduler = sched

        self.criterion = nn.CrossEntropyLoss()
        self.best = 0.0

    def on_after_backward(self):
        for i, param_group in enumerate(self.optim.param_groups):
            param_group["lr"] = self.scheduler[self.global_step]

    def forward(self, x, labels):
        with torch.no_grad():
            x = self.model.get_intermediate_layers(x, n=12)
        loss = 0
        logits = []
        for i in range(12):
            logit = self.fc[i](x[i])
            logits.append(logit)
            last = self.criterion(logit.view(-1, 1000), labels.view(-1))
            loss += last

        self.log("loss_last", last, prog_bar=True, on_step=True)

        return loss, logits

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        # print(pred_flat)
        # print(labels_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def configure_optimizers(self):
        return [self.optim]

    def training_step(self, batch, _):
        x, label = batch

        loss, logits = self.forward(x, label)

        # accuracy = self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy())

        # self.log('fine_train_loss', loss.detach().item(), on_step=True, prog_bar=True)
        # self.log('fine_train_acc', accuracy, on_step=True,
        #          prog_bar=True)
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, label = batch

        loss, logits = self.forward(x, label)

        accuracy = [self.flat_accuracy(l.detach().cpu().numpy(), label.cpu().numpy()) for l in logits]

        # self.log('fine_val_loss', loss.detach().item(), prog_bar=True)
        # self.log('fine_val_acc', prog_bar=True)

        return {'acc': accuracy}

    @torch.no_grad()
    def validation_step_end(self, batch_parts):
        # print(batch_parts)
        features = batch_parts['acc']

        return features

    @torch.no_grad()
    def validation_epoch_end(self, outs):
        accuracy = torch.tensor([f for f in outs], device=self.device) #shape[batch_size, layers, accuracy]
        gather_t = [torch.ones_like(accuracy) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, accuracy)
        accuracy = torch.cat(gather_t).to(self.device) #shape [val_samples, layers]
        for i in range(12):
            acc = accuracy[:, i].mean()
            if utils.get_rank() == 0:
                print(f"Epoch: {self.current_epoch} layer:{i+1} acc: {acc.item()}")


def expand_greyscale(t):
    return t.expand(3, -1, -1)

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

total_acc_t1 = []
total_acc_t5 = []


def main(args):
    dataset = None
    dataset_train = None
    dataset_val = None
    fine_dataset = None

    image_size = 96 if args.dataset == "stl10" else 224
    # pretrain_transform = DataAugmentationDINO(
    #     args.global_crops_scale,
    #     args.local_crops_scale,
    #     args.local_crops_number
    # )
    pretrain_transform = T.Compose([
        T.Resize((256, 256), interpolation=Image.BICUBIC),
        # T.CenterCrop(image_size),
        T.ToTensor(),
        # T.Lambda(expand_greyscale)
    ])
    fine_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = T.Compose([
        T.Resize((256, 256), interpolation=3),
        T.CenterCrop((image_size, image_size)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.dataset == "stl10":
        dataset = datasets.STL10(args.data, split='unlabeled', download=True, transform=pretrain_transform)
        dataset_train = datasets.STL10(args.data, split='train', download=True, transform=val_transform)
        dataset_val = datasets.STL10(args.data, split='test', download=True, transform=val_transform)
    elif args.dataset == "imagenet":
        # path = 'dataset'
        path = '/data/dataset/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC'
        dataset = datasets.ImageFolder(
            path + '/train',
            pretrain_transform
        )
        dataset_train = datasets.ImageFolder(
            path + '/train',
            val_transform
        )
        dataset_val = datasets.ImageFolder(
            path + '/val',
            val_transform
        )
        fine_dataset = datasets.ImageFolder(
            path + '/train',
            fine_transform
        )
    else:
        assert "error"
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(
        dataset,
        # Subset(dataset, np.arange(64)),
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    fine_loader1 = DataLoader(
        fine_dataset,
        # Subset(fine_dataset, np.arange(1024)),
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=False,
    )
    # sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    train_loader = DataLoader(
        dataset_train,
        # Subset(dataset_train, np.arange(64)),
        batch_size=args.batch_size_per_gpu,
        # sampler=sampler_train,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    # sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    val_loader = DataLoader(
        dataset_val,
        # Subset(dataset_train, np.arange(64)),
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    print("loaded dataset!")

    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    total_batch = torch.cuda.device_count() * args.batch_size_per_gpu
    clip = args.clip_grad

    args.image_size = image_size
    args.total_batch = total_batch
    args.optimizer = 'adamw'
    args.st_inter = False

    learner = PLLearner.load_from_checkpoint("/data/byol-pytorch/checkpoints/vit_small/moco_base.ckpt",
                                             student=student,
                                             teacher=teacher,
                                             length=len(data_loader),
                                             val_loader=val_loader,
                                             embed_dim=embed_dim,
                                             args=args)

    model = learner.teacher.net
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    tuner = Tuner(model, embed_dim, total_batch, len(fine_loader1), args.lr)
    fine_trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=100,
        default_root_dir="output/vit.model",
        accelerator=args.accelerator,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        sync_batchnorm=True,
        # progress_bar_refresh_rate=0
    )
    fine_trainer.fit(tuner, fine_loader1, val_loader)

    ###################################### KNN
    # model.eval()
    # model.cuda()
    #
    # train_features = []
    # train_targets = []
    # k = 20
    # retrieval_one_hot = torch.zeros(k, 1000).cuda()
    # top1, top5, total = 0.0, 0.0, 0
    #
    # for b in tqdm(train_loader):
    #     features = model(b[0].cuda()).detach()
    #     features = F.normalize(features, dim=1).cpu()
    #
    #     train_features.append(features)
    #     train_targets.append(b[1])
    #
    # train_features = torch.cat(train_features, dim=0).cuda()
    # train_targets = torch.cat(train_targets, dim=0).cuda()
    #
    # for b in tqdm(val_loader):
    #     features = model(b[0].cuda())
    #     features = F.normalize(features, dim=1).detach()
    #
    #     targets = b[1].cuda()
    #
    #     batch_size = targets.shape[0]
    #
    #     similarity = torch.mm(features, train_features.T)
    #
    #     distances, indices = similarity.topk(k, largest=True, sorted=True)
    #     distances = distances.cuda()
    #     indices = indices.cuda()
    #
    #     candidates = train_targets.view(1, -1).expand(batch_size, -1)
    #     retrieved_neighbors = torch.gather(candidates, 1, indices)
    #
    #     retrieval_one_hot.resize_(batch_size * k, 1000).zero_()
    #     retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1.0)
    #     # print("retrieval_one_hot", retrieval_one_hot)
    #     distances_transform = distances.clone().div_(0.07).exp_()
    #     # print("distances_transform", distances_transform)
    #     probs = torch.sum(
    #         torch.mul(
    #             retrieval_one_hot.view(batch_size, -1, 1000),
    #             distances_transform.view(batch_size, -1, 1),
    #         ),
    #         1,
    #     )
    #     _, predictions = probs.sort(1, True)
    #
    #     correct = predictions.eq(targets.data.view(-1, 1))
    #     top1 = top1 + correct.narrow(1, 0, 1).sum().item()
    #     top5 = top5 + correct.narrow(1, 0, 5).sum().item()
    #     total += targets.size(0)
    #
    # top1 = top1 * 100.0 / total
    # top5 = top5 * 100.0 / total
    #
    # print(f"top1: {top1}  top5: {top5}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=0.002, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=100, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=2048, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=5, help='number of workers')
    parser.add_argument('--board_path', '-bp', default='./log', type=str, help='tensorboard path')
    parser.add_argument('--accumulate', default=1, type=int, help='accumulate gradient')
    parser.add_argument('--mlp_hidden', default=4096, type=int, help='mlp hidden dimension')
    parser.add_argument('--ratio', default=1, type=int, help='loss ratio of layer2output')
    parser.add_argument('--up', default=12, type=int, help='layer2high skip layer')
    parser.add_argument('--st_inter', default=False, type=utils.bool_flag, help='intermediate representation of student')
    parser.add_argument('--t_inter', default=False, type=utils.bool_flag, help='intermediate representation of teacher')
    parser.add_argument('--l2o', default=False, type=utils.bool_flag, help='layer2output')

    parser.add_argument('--data', '-d', metavar='DIR', default='../dataset',
                        help='path to dataset')
    parser.add_argument('--dataset', '-ds', default='imagenet',
                        help='dataset name', choices=['stl10', 'cifar10', 'imagenet'])
    parser.add_argument('--name', help='name for tensorboard')
    parser.add_argument('--val_interval', default=1, type=int, help='validation epoch interval')
    parser.add_argument('--accelerator', default='ddp', type=str,
                        help='ddp for multi-gpu or node, ddp2 for across negative samples')

    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
            end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
            weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
            weight decay. We use a cosine schedule for WD and using a larger decay by
            the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
            gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
            help optimization for larger ViT architectures. 0 for disabling.""")

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_base', 'deit_tiny',
                                 'deit_small'] + torchvision_archs,
                        help="""Name of architecture to train. For quick experiments with ViTs,
                we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=32, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=256, type=int, help="""Dimensionality of
            the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--div', default=4, type=int, help="dividing hidden dimensions of mlp1")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--student', default=False, type=utils.bool_flag, help='choose student or teacher network')

    hparam = parser.parse_args()
    if hparam.load_json:
        with open(hparam.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            hparam = parser.parse_args(namespace=t_args)

    main(hparam)
