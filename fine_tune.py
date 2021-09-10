import pytorch_lightning as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import argparse
import torch.distributed as dist

from torchvision import transforms as T
from pytorch_lightning.callbacks import LearningRateMonitor
from torch import nn
import random
import numpy as np
from pl_train import PLLearner
from torchmetrics import Accuracy
from torchvision import models as torchvision_models
import vision_transformer as vits
import utils


def default(val, def_val):
    return def_val if val is None else val


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class InputMonitor(pl.Callback):
    def on_train_batch_start(self, pl_trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx % 10 == 0:
            x, y, z = batch
            sample_input = x
            sample_output = pl_module.model(sample_input.to(pl_module.device), z.to(pl_module.device)).pooler_output
            pl_logger = pl_trainer.logger
            # pl_logger.experiment.add_histogram("input", x, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("label", y, global_step=pl_trainer.global_step)
            pl_logger.experiment.add_histogram("repr_cls", sample_output, global_step=pl_trainer.global_step)
            # pl_logger.experiment.add_histogram("repr_first", sample_output[1, :], global_step=pl_trainer.global_step)


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


class Tuner(pl.LightningModule):
    def __init__(self, model, embed_dim, total_batch_size, lr=0.001):
        super().__init__()

        self.model = model
        # dim_mlp = self.model.fc[0].in_features
        # self.model.fc = nn.Identity()
        # self.model.train()
        self.fc = nn.Linear(embed_dim, 1000)

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.optim = torch.optim.SGD(
            self.fc.parameters(),
            lr * 4096 / 256.,  # linear scaling rule
            momentum=0.9,
            weight_decay=0,  # we do not apply weight decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, 100, eta_min=0)

        # self.optim = LARS(self.optim, eps=0.0)

        # sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=length, eta_min=0, last_epoch=-1)
        # w = scheduler.LinearWarmup(self.optim, warmup_steps=args.warmup, last_epoch=-1)
        # sched = scheduler.Scheduler(sched, w)
        # sched.optimizer = self.optim
        # self.scheduler = sched

        self.criterion = nn.CrossEntropyLoss()
        self.best = 0.0

    def forward(self, x, labels):
        # with torch.no_grad():
        # self.model(x)
        # x = x.unsqueeze(0)
        # x = repeat(x, '() b c -> d b c', d = 6)

        # x = self.model.layer_repr().detach()
        # x = rearrange(x, 'd b c -> b (d c)')
        x = self.model.get_representation(x)
        logits = self.fc(x)
        loss = self.criterion(logits.view(-1, 1000), labels.view(-1))

        return loss, logits

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        # print(pred_flat)
        # print(labels_flat)
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def configure_optimizers(self):
        return [self.optim], [{
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
        }]

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

        accuracy = self.flat_accuracy(logits.detach().cpu().numpy(), label.cpu().numpy())

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
        accuracy = torch.tensor([f for f in outs], device=self.device)
        gather_t = [torch.ones_like(accuracy) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_t, accuracy)
        accuracy = torch.cat(gather_t).to(self.device).mean()
        self.best = max(accuracy.item(), self.best)

        if utils.get_rank() == 0:
            print(f"Epoch: {self.current_epoch}  acc: {accuracy.item()}  best: {self.best}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vit-simclr-fine-tune')

    parser.add_argument('--checkpoint', '-ch', required=True, type=str, help='checkpoint path')
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--lr', '-l', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=100, help="epoch")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--warmup_epochs', '-w', type=int, default=300, help='warmup iteration')
    parser.add_argument('--weight-decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--accumulate', '-ac', type=int, default=1, help='gradient accumulation step')
    parser.add_argument('--num-workers', '-n', type=int, default=16, help='number of workers')
    parser.add_argument('--board-path', '-bp', default='./log', type=str, help='tensorboardx path')

    parser.add_argument('--data', '-d', metavar='DIR', default='../data',
                        help='path to dataset')
    parser.add_argument('--dataset', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10'])
    parser.add_argument('--depth', default=12, type=int, help='transformer depth')
    parser.add_argument('--name', required=True, help='name for tensorboard')
    # parser.add_argument('--checkpoint', '-ch', required=True, type=str, help='checkpoint path')

    args = parser.parse_args()
    # args.lr *= (args.batch_size / 256)

    dataset = None
    dataset_val = None

    image_size = 96 if args.dataset == "stl10" else 224
    to_tensor_transform = T.Compose(
        [T.RandomResizedCrop(image_size), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    if args.dataset == "stl10":
        dataset = datasets.STL10(args.data, split='train', download=True, transform=to_tensor_transform)
        dataset_val = datasets.STL10(args.data, split='test', download=True, transform=to_tensor_transform)
    elif args.dataset == "imagenet":
        path = '/data/data/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC'
        dataset = datasets.ImageFolder(
            path + '/train',
            to_tensor_transform
        )
        dataset_val = datasets.ImageFolder(
            path + '/val',
            to_tensor_transform
        )
    else:
        assert "error"
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=False)
    print("loaded dataset!")

    logger = pl.loggers.TensorBoardLogger(args.board_path, name='byol/fine/' + args.name)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # model = models.resnet18(pretrained=False, num_classes=128)
    # model.fc = nn.Sequential(model.fc, nn.ReLU(), nn.Linear(128, 128))

    args.patch_size = 8
    args.min_lr = 0.00001
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
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

    args.ratio = 0
    args.optimizer = "adamw"
    args.weight_decay = 0
    args.weight_decay_end = 0
    args.epochs = 1
    args.momentum_teacher = 1
    args.out_dim = 256
    args.mlp_hidden = 4096
    args.div = 1
    learner = PLLearner.load_from_checkpoint(args.checkpoint, student=student, teacher=teacher, length=0,
                                             val_loader=None, embed_dim=embed_dim, args=args)

    tuner = Tuner(learner.student.net, args, len(train_loader))
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.epoch,
        accumulate_grad_batches=args.accumulate,
        default_root_dir="output/vit.model",
        accelerator='ddp',
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[lr_monitor]
    )

    trainer.fit(tuner, train_loader, test_loader)
