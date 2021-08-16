import torch
import torch.nn.functional as F
from einops import repeat
import tqdm
import os

from torchvision import transforms as T
from torch import nn
import random
from torch.utils.data import DataLoader

from torchvision import models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import utils
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


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def expand_greyscale(t):
    return t.expand(3, -1, -1)


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def train(dataset, args):
    ratio = args.ratio
    st_inter = args.st_inter
    t_inter = args.t_inter

    clip = args.clip_grad

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    args.name = args.name + "_{}e/{}_{}_{}_{}_{}".format(args.epochs, lr, min_lr, clip, args.weight_decay,
                                                         args.weight_decay_end)

    logger = SummaryWriter(log_dir=args.name)

    # TODO collate_fn(?), dataset prepare
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = DataLoader(dataset,
                             # sampler=sampler,
                             batch_size=args.batch_size_per_gpu,
                             num_workers=args.num_workers,
                             # pin_memory=True,
                             drop_last=True,
                             )

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

    teacher.load_state_dict(student.state_dict())

    student = NetWrapper(student, embed_dim, args, prediction=True, intermediate=st_inter)
    teacher = NetWrapper(teacher, embed_dim, args, prediction=False, intermediate=t_inter)

    if st_inter != t_inter:
        teacher.projector.load_state_dict(student.projector[-1].state_dict())
    else:
        teacher.projector.load_state_dict(student.projector.state_dict())

    student, teacher = student.cuda(), teacher.cuda()

    teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[torch.cuda.current_device()],
                                                  find_unused_parameters=True)

    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Optimizer and schedulers ready.")

    aug1 = torch.nn.Sequential(
        RandomApply(
            T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        T.RandomGrayscale(p=0.2),
        T.RandomHorizontalFlip(),
        RandomApply(
            T.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        T.RandomResizedCrop((224, 224)),
        T.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])),
    )

    aug2 = aug1

    global_step = 0

    print("Start training !!")
    for epoch in tqdm(range(args.epochs)):
        for it, batch in enumerate(data_loader):
            x = batch[0].cuda()
            batch_size = x.shape[0]

            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[global_step]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[global_step]

            image_one, image_two = aug1(x), aug2(x)
            teacher_output1, student_output1, teacher_output2, student_output2 = teacher(image_one), student(
                image_two), teacher(image_two), student(image_one)

            if st_inter != t_inter:
                teacher_output1 = repeat(teacher_output1.unsqueeze(0), '() b e -> (d b) e', d=12)
                teacher_output2 = repeat(teacher_output2.unsqueeze(0), '() b e -> (d b) e', d=12)

            if ratio > 0:
                student_mid1, student_output1 = torch.split(student_output1, [batch_size * 11, batch_size], dim=0)
                student_mid2, student_output2 = torch.split(student_output2, [batch_size * 11, batch_size], dim=0)
                teacher_mid1, teacher_output1 = torch.split(teacher_output1, [batch_size * 11, batch_size], dim=0)
                teacher_mid2, teacher_output2 = torch.split(teacher_output2, [batch_size * 11, batch_size], dim=0)
                loss_mid = loss_fn(student_mid1, teacher_mid1).mean() + loss_fn(student_mid2, teacher_mid2).mean()
                loss_output = loss_fn(student_output1, teacher_output1).mean() + loss_fn(student_output2,
                                                                                         teacher_output2).mean()
                loss = loss_output + ratio * loss_mid
            else:
                loss = loss_fn(student_output1, teacher_output1).mean()
                loss += loss_fn(student_output2, teacher_output2).mean()
                if st_inter:
                    loss *= 12

            optimizer.zero_grad()
            loss.backward()
            param_norms = utils.clip_gradients(student, clip)
            optimizer.step()

            # EMA
            with torch.no_grad():
                m = momentum_schedule[global_step]
                for current_params, ma_params in zip(student.net.parameters(), teacher_without_ddp.net.parameters()):
                    old_weight, up_weight = ma_params.data, current_params.data
                    ma_params.data = old_weight * m + (1 - m) * up_weight

                if st_inter != t_inter:
                    for current_params, ma_params in zip(student.projector[-1].parameters(),
                                                         teacher_without_ddp.projector.parameters()):
                        old_weight, up_weight = ma_params.data, current_params.data
                        ma_params.data = old_weight * m + (1 - m) * up_weight
                else:
                    for current_params, ma_params in zip(student.projector.parameters(),
                                                         teacher_without_ddp.projector.parameters()):
                        old_weight, up_weight = ma_params.data, current_params.data
                        ma_params.data = old_weight * m + (1 - m) * up_weight

            # logging
            logger.add_scalar('loss', loss.item(), global_step)
            logger.add_scalar('lr', optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar('wd', optimizer.param_groups[0]["weight_decay"], global_step)

            global_step += 1

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }

        utils.save_on_master(save_dict, os.path.join(args.name, f'checkpoint.pth'))
        if epoch % 10 == 0:
            utils.save_on_master(save_dict, os.path.join(args.name, f'checkpoint{epoch:04}.pth'))
