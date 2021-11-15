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
import pickle
from urllib.request import urlopen
from einops import repeat
from PIL import Image, ImageFile

from torchvision import transforms as T
from torchvision import models as torchvision_models

import utils
import json
import numpy as np
import vision_transformer as vits

from PIL import Image
from pl_train_moco import PLLearner


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False, n=1):
    features = None
    for samples, index in data_loader:
        _, _, h, w = samples.shape

        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        if multiscale:
            feats = utils.multi_scale(samples, model, n=n)
        else:
            feats = model.get_intermediate_layers_all(samples, n=n)[0].clone()
            feats = feats[:, 0, :]

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


class OxfordParisDataset(torch.utils.data.Dataset):
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def config_imname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


def main(args):
    utils.init_distributed_mode(args)
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

    # model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    total_batch = torch.cuda.device_count() * args.batch_size_per_gpu
    clip = args.clip_grad

    args.image_size = 224
    args.total_batch = total_batch
    args.optimizer = 'adamw'
    args.temperature = 0.2

    learner = PLLearner.load_from_checkpoint(args.ckpt,
                                             student=student,
                                             teacher=teacher,
                                             length=0,
                                             val_loader=None,
                                             embed_dim=embed_dim,
                                             args=args)
    model = learner.teacher.net

    # model = model.net

    # model.load_state_dict(torch.hub.load_state_dict_from_url(
    #     url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth"))

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.cuda()


    # ============ Extract features ... ============
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = OxfordParisDataset(args.data_path, args.dataset, split="train", transform=transform,
                                       imsize=args.imsize)
    dataset_query = OxfordParisDataset(args.data_path, args.dataset, split="query", transform=transform,
                                       imsize=args.imsize)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"train: {len(dataset_train)} imgs / query: {len(dataset_query)} imgs")

    train_features = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale, n=args.n)
    query_features = extract_features(model, data_loader_query, args.use_cuda, multiscale=args.multiscale, n=args.n)

    train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
    query_features = torch.nn.functional.normalize(query_features, dim=1, p=2)

    ############################################################################
    # Step 2: similarity
    sim = torch.mm(train_features, query_features.T)
    ranks = torch.argsort(-sim, dim=0).cpu().numpy()

    ############################################################################
    # Step 3: evaluate
    gnd = dataset_train.cfg['gnd']
    # evaluate ranks
    ks = [1, 5, 10]
    # search for easy & hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_t.append(g)
    mapM, apsM, mprM, prsM = utils.compute_map(ranks, gnd_t, ks)
    # search for hard
    gnd_t = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_t.append(g)
    mapH, apsH, mprH, prsH = utils.compute_map(ranks, gnd_t, ks)
    print('>> {}: mAP M: {}, H: {}'.format(args.dataset, np.around(mapM * 100, decimals=2),
                                           np.around(mapH * 100, decimals=2)))
    print('>> {}: mP@k{} M: {}, H: {}'.format(args.dataset, np.array(ks), np.around(mprM * 100, decimals=2),
                                              np.around(mprH * 100, decimals=2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help="epochs for scheduling")
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

    parser.add_argument('--data_path', default='/dataset/revisitop/data/datasets/', type=str)
    parser.add_argument('--dataset', default='rparis6k', type=str, choices=['roxford5k', 'rparis6k'])
    parser.add_argument('--multiscale', default=True, type=utils.bool_flag)
    parser.add_argument('--imsize', default=512, type=int, help='Image size')
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--ckpt", type=str, help="ckpt")
    parser.add_argument("--n", type=int, help="layer")

    hparam = parser.parse_args()
    if hparam.load_json:
        with open(hparam.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            hparam = parser.parse_args(namespace=t_args)

    main(hparam)