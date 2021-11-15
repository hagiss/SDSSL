import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
from tqdm import tqdm
import copy
import torch.nn.functional as F

from torchvision import transforms as T
from torchvision import models as torchvision_models

import utils
import json
import numpy as np
import vision_transformer as vits

from PIL import Image
from pl_train_simclr import PLLearner


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
            pretrain_transform
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
        pin_memory=True,
    )
    # fine_loader1 = DataLoader(
    #     fine_dataset,
    #     # Subset(fine_dataset, np.arange(1024)),
    #     batch_size=args.batch_size_per_gpu,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=True,
    #     pin_memory=True,
    # )
    # sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    train_loader = DataLoader(
        dataset_train,
        # Subset(dataset_train, np.arange(64)),
        batch_size=args.batch_size_per_gpu,
        # sampler=sampler_train,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False)
    val_loader = DataLoader(
        # dataset_val,
        Subset(dataset_val, np.arange(1000)),
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("loaded dataset!")

    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            # drop_path_rate=0.1,  # stochastic depth
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

    # student = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)
    # teacher = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    total_batch = torch.cuda.device_count() * args.batch_size_per_gpu
    clip = args.clip_grad

    args.image_size = image_size
    args.total_batch = total_batch
    args.optimizer = 'adamw'
    args.temperature = 0.2
    args.student = True

    args.st_inter = False

    learner_base = PLLearner.load_from_checkpoint("/data/byol-pytorch/checkpoints/vit_small/simclr_base.ckpt",
                                             student=student,
                                             # teacher=teacher,
                                             length=len(data_loader),
                                             val_loader=val_loader,
                                             embed_dim=embed_dim,
                                             args=args)
    aug1 = learner_base.aug1.cuda()
    aug2 = learner_base.aug2.cuda()
    # for i in val_loader:
    #     img1 = i[0].cuda()
    #     img1 = aug1(img1)
    #     break
    model_base = copy.deepcopy(learner_base.student) if args.student else copy.deepcopy(learner_base.teacher)

    args.st_inter = True
    for p in model_base.parameters():
        p.requires_grad = False

    model_base.eval()
    model_base.cuda()

    # print(model_base.get_representation(img1, intermediate=True).cpu()[0, :5])

    learner_l2o = PLLearner.load_from_checkpoint("/data/byol-pytorch/checkpoints/vit_small/simclr_l2o.ckpt",
                                             student=student,
                                             # teacher=teacher,
                                             length=len(data_loader),
                                             val_loader=val_loader,
                                             embed_dim=embed_dim,
                                             args=args)

    model_l2o = learner_l2o.student if args.student else learner_l2o.teacher

    for p in model_l2o.parameters():
        p.requires_grad = False
    model_l2o.eval()
    model_l2o.cuda()

    # print(model_l2o.get_representation(img1, intermediate=True).cpu()[0, :5])

    features1 = [[] for _ in range(12)]
    features2 = [[] for _ in range(12)]

    features1_l2o = [[] for _ in range(12)]
    features2_l2o = [[] for _ in range(12)]

    for b in tqdm(val_loader):
        img = b[0].cuda()
        img1 = aug2(img)
        img2 = aug2(img)
        batch_size = img1.shape[0]

        rep1 = model_base.get_representation(img1, intermediate=True).cpu()
        rep2 = model_base.get_representation(img2, intermediate=True).cpu()

        rep1_l2o = model_l2o.get_representation(img1, intermediate=True).cpu()
        rep2_l2o = model_l2o.get_representation(img2, intermediate=True).cpu()

        # rep1 = model(img1)
        # rep2 = model(img2)
        rep1 = F.normalize(rep1, dim=-1, p=2).cpu()
        rep2 = F.normalize(rep2, dim=-1, p=2).cpu()

        rep1_l2o = F.normalize(rep1_l2o, dim=-1, p=2).cpu()
        rep2_l2o = F.normalize(rep2_l2o, dim=-1, p=2).cpu()

        for i in range(12):
            features1[i].append(rep1[i*batch_size:(i+1)*batch_size, :])
            features2[i].append(rep2[i*batch_size:(i+1)*batch_size, :])

            features1_l2o[i].append(rep1_l2o[i * batch_size:(i + 1) * batch_size, :])
            features2_l2o[i].append(rep2_l2o[i * batch_size:(i + 1) * batch_size, :])

    alignment = []
    # alignment_last = []
    uniformity = []

    alignment_l2o = []
    # alignment_last_l2o = []
    uniformity_l2o = []

    ratio = []
    ratio_l2o = []

    # last_features1 = torch.cat(features1[11], dim=0).cuda()
    # last_features2 = torch.cat(features2[11], dim=0).cuda()

    # last_features1_l2o = torch.cat(features1_l2o[11], dim=0).cuda()
    # last_features2_l2o = torch.cat(features2_l2o[11], dim=0).cuda()

    # print(last_features1[0, :5])
    # print(last_features1_l2o[0, :5])

    for _ in tqdm(range(12)):
        i = 0
        features1[i] = torch.cat(features1[i], dim=0).cuda()
        features2[i] = torch.cat(features2[i], dim=0).cuda()

        features1_l2o[i] = torch.cat(features1_l2o[i], dim=0).cuda()
        features2_l2o[i] = torch.cat(features2_l2o[i], dim=0).cuda()

        # align = torch.pow(torch.norm(features1[i] - last_features2, dim=1), 2).mean().item()
        # align += torch.pow(torch.norm(features2[i] - last_features1, dim=1), 2).mean().item()
        # align /= 2
        # alignment_last.append(align)
        align = torch.pow(torch.norm(features1[i] - features2[i], dim=1), 2).mean().item()
        alignment.append(align)

        sq_pdist = torch.pow(torch.pdist(features1[i], p=2), 2)
        # uni = -sq_pdist.mul(-2).exp().mean().log().item()
        uni = sq_pdist.mean().item()
        uniformity.append(uni)

        ratio.append(uni/align)

        del features1[i]
        del features2[i]

        # l2o
        # align_l2o = torch.pow(torch.norm(features1_l2o[i] - last_features2_l2o, dim=1), 2).mean().item()
        # align_l2o += torch.pow(torch.norm(features2_l2o[i] - last_features1_l2o, dim=1), 2).mean().item()
        # align_l2o /= 2
        # alignment_last_l2o.append(align_l2o)
        align_l2o = torch.pow(torch.norm(features1_l2o[i] - features2_l2o[i], dim=1), 2).mean().item()
        alignment_l2o.append(align_l2o)

        sq_pdist_l2o = torch.pow(torch.pdist(features1_l2o[i], p=2), 2)
        # uni_l2o = -sq_pdist_l2o.mul(-2).exp().mean().log().item()
        uni_l2o = sq_pdist_l2o.mean().item()
        uniformity_l2o.append(uni_l2o)

        ratio_l2o.append(uni_l2o / align_l2o)

        del features1_l2o[i]
        del features2_l2o[i]

    r = 6
    print('alignment', np.round(alignment, r))
    print('alignment_l2o', np.round(alignment_l2o, r))

    # print('alignment_last', np.round(alignment_last, r))
    # print('alignment_last_l2o', np.round(alignment_last_l2o, r))

    print('uniformity', np.round(uniformity, r))
    print('uniformity_l2o', np.round(uniformity_l2o, r))

    print('ratio', np.round(ratio, r))
    print('ratio_l2o', np.round(ratio_l2o, r))


    # for i in range(12):
        # features1[i] = torch.cat(features1[i], dim=0)
        # features2[i] = torch.cat(features2[i], dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=300, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=512, help="batch size")
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