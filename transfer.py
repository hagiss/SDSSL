import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
from tqdm import tqdm
import copy
import torch.nn.functional as F
import torch.distributed as dist
import os
import queue
import cv2
import glob
from urllib.request import urlopen
from einops import repeat

from torchvision import transforms as T
from torchvision import models as torchvision_models

from timm.data import Mixup
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma
import utils
import json
import numpy as np
import vision_transformer as vits

from PIL import Image
from pl_train_moco import PLLearner as MOCO
# from pl_train import PLLearner as BYOL
# from pl_train_simclr import PLLearner as SimCLR

from datasets import build_dataset
import pytorch_lightning as pl


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


class PLLearner(pl.LightningModule):
    def __init__(self, model, class_num, args):
        super().__init__()
        self.model = torch.nn.Sequential(model, torch.nn.Linear(args.embed_dim, args.nb_classes))

        self.mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            self.mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=class_num)

        args.lr = args.lr * args.batch_size_per_gpu * utils.get_world_size() / 512.0

        self.optimizer = create_optimizer(args, self.model)
        self.lr_scheduler, _ = create_scheduler(args, self.optimizer)

        self.criterion = LabelSmoothingCrossEntropy()
        self.best = 0.0

        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            self.criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return [self.optimizer]

    def training_step(self, batch, batch_idx):
        samples, targets = batch

        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)

        feature = self.model(samples)
        loss = self.criterion(feature, targets) * 0.5

        return {'loss': loss}

    def training_epoch_end(self, _):
        self.lr_scheduler.step(self.current_epoch)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, label = batch

        logits = self.model(x)

        accuracy = self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy())

        return {'acc': accuracy}

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    @torch.no_grad()
    def validation_step_end(self, batch_parts):
        features = batch_parts['acc']

        return features

    @torch.no_grad()
    def validation_epoch_end(self, outs):
        accuracy = torch.tensor([f for f in outs], device=self.device)
        gather_t = [torch.ones_like(accuracy) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, accuracy)
        accuracy = torch.cat(gather_t).to(self.device).mean()
        self.best = max(accuracy.item(), self.best)

        if utils.get_rank() == 0:
            print(f"Epoch: {self.current_epoch}  acc: {accuracy.item()}  best: {self.best}")

def main(args):

    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_rate=0,
            drop_path_rate=0,  # stochastic depth
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
    #
    # # student = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)
    # # teacher = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)

    args.image_size = 224
    args.optimizer = 'adamw'
    args.total_batch = 1024
    args.weight_decay_end = 0.1

    args.st_inter = True

    learner = MOCO.load_from_checkpoint("/data/byol-pytorch/checkpoints/vit_small/moco_l2o_6.ckpt",
                                             student=student,
                                             teacher=teacher,
                                             length=0,
                                             val_loader=None,
                                             embed_dim=embed_dim,
                                             args=args)
    model = learner.teacher
    model = model.net

    for p in model.parameters():
        p.requires_grad = True

    #####################################
    args.input_size = 224
    args.embed_dim = embed_dim
    args.data_path = "../data/pets/"

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size_per_gpu),
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False
    )

    learner = PLLearner(model, args.nb_classes, args)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=100,
        default_root_dir="output/vit.model",
        accelerator="ddp",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        sync_batchnorm=True,
    )
    trainer.fit(learner, data_loader_train, data_loader_val)


    ###################################### KNN
    # model.eval()
    # model.cuda()
    #
    # train_features = []
    # train_targets = []
    # k = 20
    # retrieval_one_hot = torch.zeros(k, args.nb_classes).cuda()
    # top1, top5, total = 0.0, 0.0, 0
    #
    # for b in tqdm(data_loader_train):
    #     features = model(b[0].cuda()).cpu()
    #     features = F.normalize(features, dim=1).cpu()
    #
    #     train_features.append(features)
    #     train_targets.append(b[1])
    #
    # train_features = torch.cat(train_features, dim=0).cuda()
    # train_targets = torch.cat(train_targets, dim=0).cuda()
    #
    # for b in tqdm(data_loader_val):
    #     features = model(b[0].cuda())
    #     features = F.normalize(features, dim=1)
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
    #     retrieval_one_hot.resize_(batch_size * k, args.nb_classes).zero_()
    #     retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1.0)
    #     # print("retrieval_one_hot", retrieval_one_hot)
    #     distances_transform = distances.clone().div_(0.03).exp_()
    #     # print("distances_transform", distances_transform)
    #     probs = torch.sum(
    #         torch.mul(
    #             retrieval_one_hot.view(batch_size, -1, args.nb_classes),
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

    parser.add_argument('--epochs', '-e', type=int, default=100, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=100, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=10, help='number of workers')
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
    parser.add_argument('--data-set', '-ds', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10', 'cifar100', 'imagenet', 'flowers', 'pets'])
    parser.add_argument('--name', help='name for tensorboard')
    parser.add_argument('--val_interval', default=10, type=int, help='validation epoch interval')
    parser.add_argument('--accelerator', default='ddp', type=str,
                        help='ddp for multi-gpu or node, ddp2 for across negative samples')

    parser.add_argument("--warmup-epochs", default=4, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                                 "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

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