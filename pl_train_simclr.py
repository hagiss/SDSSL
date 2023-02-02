import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import argparse
import torch.nn.functional as F
from einops import repeat

from torchvision import transforms as T
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
import random
import torch.distributed as dist
from torchvision import models as torchvision_models

import utils
import json
import math
import vision_transformer as vits
from byol_pytorch import NetWrapper

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


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return (2 - 2 * (x * y).sum(dim=-1)).mean()


class PLLearner(pl.LightningModule):
    def __init__(self, student, length, val_loader, embed_dim, args):
        super().__init__()
        self.ratio = args.ratio
        self.st_inter = args.st_inter
        self.t_inter = args.t_inter
        self.temperature = args.temperature

        self.student = NetWrapper(student, embed_dim, args, prediction=False, intermediate=self.st_inter)

        print(f"Student is built: {args.arch} network.")

        # ============ preparing optimizer ... ============
        params_groups = utils.get_params_groups(self.student)
        if args.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
        elif args.optimizer == "lars":
            self.optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

        length = math.ceil(length / (args.accumulate * torch.cuda.device_count()))

        # ============ init schedulers ... ============
        self.lr_schedule = utils.cosine_scheduler(
            args.lr * args.total_batch / 256.,
            args.min_lr,
            args.epochs, length,
            warmup_epochs=args.warmup_epochs,
        )
        self.wd_schedule = utils.cosine_scheduler(
            args.weight_decay,
            args.weight_decay_end,
            args.epochs, length,
        )
        self.ratio_schedule = utils.cosine_scheduler(
            0, args.ratio,
            args.epochs, length,
        )

        print(f"Loss, optimizer and schedulers ready.")

        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.automatic_optimization = False

        self.labels = None
        self.label = None
        self.mask = None

        self.labels_int = None
        self.label_int = None
        self.mask_int = None

    def configure_optimizers(self):
        return [self.optimizer]

    def forward(self, x1, x2):
        return self.student(torch.cat((x1, x2), dim=0))

    def info_nce_loss(self, features):
        b, _ = features.shape
        if self.labels is None:
            self.labels = torch.cat([torch.arange(b/2) for i in range(2)], dim=0)
            self.labels = (self.labels.unsqueeze(0) == self.labels.unsqueeze(1)).float()
            temp = [torch.zeros(self.labels.shape) for _ in range(torch.distributed.get_world_size())]
            temp[torch.distributed.get_rank()] = self.labels
            self.labels = torch.cat(temp, dim=0).T
            self.labels.to(self.device)

        output = torch.cat(GatherLayer.apply(features), dim=0)

        similarity_matrix = torch.matmul(features, output.T)

        if self.mask is None:
            self.mask = torch.eye(b, dtype=torch.bool)
            temp = [torch.zeros(self.mask.shape, dtype=torch.bool) for _ in
                    range(torch.distributed.get_world_size())]
            temp[torch.distributed.get_rank()] = self.mask
            self.mask = torch.cat(temp, dim=0).to(self.device).T
        labels = self.labels[~self.mask].view(self.labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~self.mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        if self.label is None:
            self.label = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        logits = logits / 0.2
        return self.criterion(logits, self.label)

    def info_nce_loss_intermediate(self, layer_features, output):
        b = layer_features.shape[0]

        output = concat_all_gather(output)
        similarity_matrix = torch.matmul(layer_features, output.T)

        if self.labels_int is None:
            self.labels_int = torch.cat([torch.arange(b/22) for i in range(2)], dim=0)
            self.labels_int = (self.labels_int.unsqueeze(0) == self.labels_int.unsqueeze(1)).float()
            temp = [torch.zeros(self.labels_int.shape) for _ in range(torch.distributed.get_world_size())]
            temp[torch.distributed.get_rank()] = self.labels_int
            self.labels_int = torch.cat(temp, dim=0).T
            self.labels_int = repeat(self.labels_int, "h w -> (r h) w", r=11).to(self.device)

        if self.mask_int is None:
            self.mask_int = torch.eye(int(b/11), dtype=torch.bool)
            temp = [torch.zeros(self.mask_int.shape, dtype=torch.bool) for _ in range(torch.distributed.get_world_size())]
            temp[torch.distributed.get_rank()] = self.mask_int
            self.mask_int = torch.cat(temp, dim=0).to(self.device).T
            self.mask_int = repeat(self.mask_int, "h w -> (r h) w", r=11)

        labels = self.labels_int[~self.mask_int].view(self.labels_int.shape[0], -1)
        similarity_matrix = similarity_matrix[~self.mask_int].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        if self.label_int is None:
            self.label_int = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)

        logits = logits / 0.2
        return self.criterion(logits, self.label_int)

    def training_step(self, batch, batch_idx):
        images1, images2 = batch[0][0], batch[0][1]
        batch_size = images1.shape[0] * 2

        self.update_lr()

        student_output = F.normalize(self.forward(images1, images2), dim=1)

        loss_mid = 0.0
        if self.st_inter != self.t_inter:
            student_mid, student_output = torch.split(student_output, [batch_size * 11, batch_size], dim=0)
            loss_mid = self.info_nce_loss_intermediate(student_mid, student_output.detach())
        loss_output = self.info_nce_loss(student_output)

        loss = loss_output + self.ratio * loss_mid

        opt = self.optimizer
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.logger.experiment.add_scalar('loss', loss.detach().item(), self.global_step)

        return {'loss': loss}

    def update_lr(self):
        self.ratio = self.ratio_schedule[self.global_step]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[self.global_step]
            if i == 0:
                self.logger.experiment.add_scalar('lr', self.lr_schedule[self.global_step], self.global_step)
                param_group["weight_decay"] = self.wd_schedule[self.global_step]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, label = batch

        features = self.student.get_representation(x).detach().cpu()
        return {'features': features, 'labels': label}

    @torch.no_grad()
    def validation_step_end(self, batch_parts):
        # print(batch_parts)
        features = batch_parts['features']
        labels = batch_parts['labels']

        return features, labels

    @torch.no_grad()
    def validation_epoch_end(self, outs):
        train_features = torch.cat([f[0] for f in outs]).to(self.device)
        gather_t = [torch.ones_like(train_features) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, train_features)
        train_features = torch.cat(gather_t).to(self.device)
        train_features = F.normalize(train_features, dim=1).t()

        train_labels = torch.cat([f[1] for f in outs])
        gather_t = [torch.ones_like(train_labels) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, train_labels)
        train_labels = torch.cat(gather_t).to(self.device)

        k = 20
        num_classes = 1000
        retrieval_one_hot = torch.zeros(k, num_classes).to(self.device)
        top1, top5, total = 0.0, 0.0, 0
        # print("train_features", train_features)
        # print(len(self.val_loader))

        for batch in self.val_loader:
            features = self.student.get_representation(batch[0].to(self.device))
            features = F.normalize(features, dim=1)
            # print("features", features)
            targets = batch[1].to(self.device)
            # print(targets)

            batch_size = targets.shape[0]

            similarity = torch.mm(features, train_features)
            # print("similarity", similarity)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            distances = distances.to(self.device)
            indices = indices.to(self.device)
            # print("distances", distances)
            # print("indices", indices)
            candidates = train_labels.view(1, -1).expand(batch_size, -1)
            # print("candidates", candidates)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1.0)
            # print("retrieval_one_hot", retrieval_one_hot)
            distances_transform = distances.clone().div_(0.07).exp_()
            # print("distances_transform", distances_transform)
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            # print("probs", probs)
            _, predictions = probs.sort(1, True)
            # print("prediction", predictions)

            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        # print(top1, top5)
        if utils.get_rank() == 0:
            print(f"Epoch: {self.current_epoch}  top1: {top1}  top5: {top5}")
            total_acc_t1.append(top1)
            total_acc_t5.append(top5)
        self.logger.experiment.add_scalar('top1', top1, self.current_epoch)
        self.logger.experiment.add_scalar('top5', top5, self.current_epoch)


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

    image_size = 96 if args.dataset == "stl10" else 224
    image_size_before_crop = 96 if args.dataset == "stl10" else 256

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    augmentation1 = [
        T.RandomResizedCrop(image_size, scale=(0.08, 1.)),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([utils.GaussianBlur([.1, 2.])], p=1.0),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]

    augmentation2 = [
        T.RandomResizedCrop(image_size, scale=(0.08, 1.)),
        T.RandomApply([
            T.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.1),
        T.RandomApply([utils.Solarize()], p=0.2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ]

    pretrain_transform = utils.TwoCropsTransform(T.Compose(augmentation1), T.Compose(augmentation2))

    val_transform = T.Compose([
        T.Resize((image_size_before_crop, image_size_before_crop), interpolation=3),
        T.CenterCrop((image_size, image_size)),
        T.ToTensor(),
        normalize,
    ])

    if args.dataset == "stl10":
        dataset = datasets.STL10(args.data, split='unlabeled', download=True, transform=pretrain_transform)
        dataset_train = datasets.STL10(args.data, split='train', download=True, transform=val_transform)
        dataset_val = datasets.STL10(args.data, split='test', download=True, transform=val_transform)
    elif args.dataset == "imagenet":
        # path = 'dataset'
        # path = '/data/dataset/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC'
        path = args.data
        dataset = datasets.ImageFolder(
            path + '/train',
            pretrain_transform
        )
        dataset_train = datasets.ImageFolder(          # for knn evaluation
            path + '/train',
            val_transform
        )
        dataset_val = datasets.ImageFolder(
            path + '/val',
            val_transform
        )
    else:
        assert "error"
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("loaded dataset!")

    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            img_size=[image_size],
            patch_size=args.patch_size,
            dis_token=args.dis_token,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    total_batch = torch.cuda.device_count() * args.accumulate * args.batch_size_per_gpu
    clip = args.clip_grad

    args.image_size = image_size
    args.total_batch = total_batch

    learner = PLLearner(student, len(data_loader), val_loader, embed_dim, args)

    logger = pl.loggers.TensorBoardLogger(args.board_path, name=args.name + "_{}e/{}_{}_{}_{}_{}_{}".format(args.epochs, lr, min_lr, total_batch, clip, args.weight_decay, args.weight_decay_end))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.max_epochs,
        default_root_dir="output/vit.model",
        accelerator=args.accelerator,
        logger=logger,
        num_sanity_val_steps=0,
        gradient_clip_val=args.clip_grad,
        accumulate_grad_batches=args.accumulate,
        check_val_every_n_epoch=args.val_interval,
        sync_batchnorm=True,
        callbacks=[lr_monitor],
        progress_bar_refresh_rate=1
    )

    trainer.fit(learner, data_loader, train_loader)

    if utils.get_rank() == 0:
        print("top1", total_acc_t1)
        print("best top1", max(total_acc_t1))
        print("top5", total_acc_t5)
        print("best top5", max(total_acc_t5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=300, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=10, help='number of workers')
    parser.add_argument('--board_path', '-bp', default='./log', type=str, help='tensorboard path')
    parser.add_argument('--accumulate', default=1, type=int, help='accumulate gradient')
    parser.add_argument('--mlp_hidden', default=4096, type=int, help='mlp hidden dimension')
    parser.add_argument('--ratio', default=1, type=float, help='loss ratio of self-distillation')
    parser.add_argument('--st_inter', default=False, type=bool, help='apply self-distillation')

    parser.add_argument('--data', '-d', metavar='DIR', default='../dataset',
                        help='path to dataset')
    parser.add_argument('--dataset', '-ds', default='stl10',
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
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
            of input square patches - default 16 (for 16x16 patches). Using smaller
            values leads to better performance but requires more memory. Applies only
            for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
            mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=512, type=int, help="""Dimensionality of
            the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
            parameter for teacher update. The value is increased to 1 during training with cosine schedule.
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--dis_token', default=False, type=utils.bool_flag, help="distillation token")

    hparam = parser.parse_args()
    if hparam.load_json:
        with open(hparam.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            hparam = parser.parse_args(namespace=t_args)

    main(hparam)

