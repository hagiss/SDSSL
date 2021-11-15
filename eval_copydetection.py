import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
from tqdm import tqdm
import copy
import torch.nn.functional as F
import torch.distributed as dist
import os

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

total_acc_t1 = []
total_acc_t5 = []

class CopydaysDataset():
    def __init__(self, basedir):
        self.basedir = basedir
        self.block_names = (
            ['original', 'strong'] +
            ['jpegqual/%d' % i for i in
             [3, 5, 8, 10, 15, 20, 30, 50, 75]] +
            ['crops/%d' % i for i in
             [10, 15, 20, 30, 40, 50, 60, 70, 80]])
        self.nblocks = len(self.block_names)

        self.query_blocks = range(self.nblocks)
        self.q_block_sizes = np.ones(self.nblocks, dtype=int) * 157
        self.q_block_sizes[1] = 229
        # search only among originals
        self.database_blocks = [0]

    def get_block(self, i):
        dirname = self.basedir + '/' + self.block_names[i]
        fnames = [dirname + '/' + fname
                  for fname in sorted(os.listdir(dirname))
                  if fname.endswith('.jpg')]
        return fnames

    def get_block_filenames(self, subdir_name):
        dirname = self.basedir + '/' + subdir_name
        return [fname
                for fname in sorted(os.listdir(dirname))
                if fname.endswith('.jpg')]

    def eval_result(self, ids, distances):
        j0 = 0
        for i in range(self.nblocks):
            j1 = j0 + self.q_block_sizes[i]
            block_name = self.block_names[i]
            I = ids[j0:j1]   # block size
            sum_AP = 0
            if block_name != 'strong':
                # 1:1 mapping of files to names
                positives_per_query = [[i] for i in range(j1 - j0)]
            else:
                originals = self.get_block_filenames('original')
                strongs = self.get_block_filenames('strong')

                # check if prefixes match
                positives_per_query = [
                    [j for j, bname in enumerate(originals)
                     if bname[:4] == qname[:4]]
                    for qname in strongs]

            for qno, Iline in enumerate(I):
                positives = positives_per_query[qno]
                ranks = []
                for rank, bno in enumerate(Iline):
                    if bno in positives:
                        ranks.append(rank)
                sum_AP += score_ap_from_ranks_1(ranks, len(positives))

            print("eval on %s mAP=%.3f" % (
                block_name, sum_AP / (j1 - j0)))
            j0 = j1


# from the Holidays evaluation package
def score_ap_from_ranks_1(ranks, nres):
    """ Compute the average precision of one search.
    ranks = ordered list of ranks of true positives
    nres  = total number of positives in dataset
    """

    # accumulate trapezoids in PR-plot
    ap = 0.0

    # All have an x-size of:
    recall_step = 1.0 / nres

    for ntp, rank in enumerate(ranks):

        # y-size on left side of trapezoid:
        # ntp = nb of true positives so far
        # rank = nb of retrieved items so far
        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = ntp / float(rank)

        # y-size on right side of trapezoid:
        # ntp and rank are increased by one
        precision_1 = (ntp + 1) / float(rank + 1)

        ap += (precision_1 + precision_0) * recall_step / 2.0

    return ap


class ImgListDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, transform=None):
        self.samples = img_list
        self.transform = transform

    def __getitem__(self, i):
        with open(self.samples[i], 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, i

    def __len__(self):
        return len(self.samples)


def is_image_file(s):
    ext = s.split(".")[-1]
    if ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp']:
        return True
    return False


@torch.no_grad()
def extract_features(image_list, model, args):
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size), interpolation=3),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tempdataset = ImgListDataset(image_list, transform=transform)
    data_loader = torch.utils.data.DataLoader(tempdataset, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, drop_last=False,
        sampler=torch.utils.data.DistributedSampler(tempdataset, shuffle=False))
    features = None
    for samples, index in data_loader:
        samples, index = samples.cuda(non_blocking=True), index.cuda(non_blocking=True)
        feats = model.get_intermediate_layers_all(samples, n=args.n)

        # print(feats.shape)

        cls_output_token = feats[0][:, 0, :]  #  [CLS] token
        # cls_output_token = cls_output_token.unsqueeze(0)
        # GeM with exponent 4 for output patch tokens
        # print(feats[0][:, 1:, :].shape)
        # print(samples.shape[-2])

        b, h, w, d = len(samples), int(samples.shape[-2] / model.patch_embed.patch_size[0]), int(samples.shape[-1] / model.patch_embed.patch_size[1]), feats[0].shape[-1]
        # print(b, h, w, d)
        feats = feats[0][:, 1:, :].reshape(b, h, w, d)
        # feats = feats.unsqueeze(0).reshape(b, h, w, d)
        feats = feats.clamp(min=1e-6).permute(0, 3, 1, 2)
        feats = torch.nn.functional.avg_pool2d(feats.pow(4), (h, w)).pow(1. / 4).reshape(b, -1)
        # concatenate [CLS] token and GeM pooled patch tokens
        feats = torch.cat((cls_output_token, feats), dim=1)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            features = features.cuda(non_blocking=True)

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1),
                                dtype=feats.dtype, device=feats.device)
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(output_l))
    return features  # features is still None for every rank which is not 0 (main)


def main(args):

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

    utils.init_distributed_mode(args)

    args.image_size = 224
    args.total_batch = 100
    args.optimizer = 'adamw'
    args.temperature = 0.2

    learner = PLLearner.load_from_checkpoint(args.ckpt,
                                             student=student,
                                             teacher=teacher,
                                             length=0,
                                             val_loader=None,
                                             embed_dim=embed_dim,
                                             args=args)
    model = learner.teacher

    model = model.net

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.cuda()

    dataset = CopydaysDataset(args.data_path)

    # ============ Extract features ... ============
    # extract features for queries
    queries = []
    for q in dataset.query_blocks:
        queries.append(extract_features(dataset.get_block(q), model, args))
    if utils.get_rank() == 0:
        queries = torch.cat(queries)
        print(f"Extraction of queries features done. Shape: {queries.shape}")

    # extract features for database
    database = []
    for b in dataset.database_blocks:
        database.append(extract_features(dataset.get_block(b), model, args))

    # extract features for distractors
    if os.path.isdir(args.distractors_path):
        print("Using distractors...")
        list_distractors = [os.path.join(args.distractors_path, s) for s in os.listdir(args.distractors_path)]
        database.append(extract_features(list_distractors, model, args))
    if utils.get_rank() == 0:
        database = torch.cat(database)
        print(f"Extraction of database and distractors features done. Shape: {database.shape}")

    # ============ Whitening ... ============
    if os.path.isdir(args.whitening_path):
        print(f"Extracting features on images from {args.whitening_path} for learning the whitening operator.")
        list_whit = [os.path.join(args.whitening_path, s) for s in os.listdir(args.whitening_path)]
        # print(len(list_whit))
        features_for_whitening = extract_features(list_whit, model, args)
        if utils.get_rank() == 0:
            # center
            mean_feature = torch.mean(features_for_whitening, dim=0)
            database -= mean_feature
            queries -= mean_feature
            pca = utils.PCA(dim=database.shape[-1], whit=0.5)
            # compute covariance
            cov = torch.mm(features_for_whitening.T, features_for_whitening) / features_for_whitening.shape[0]
            pca.train_pca(cov.cpu().numpy())
            database = pca.apply(database)
            queries = pca.apply(queries)

    # ============ Copy detection ... ============
    if utils.get_rank() == 0:
        # l2 normalize the features
        database = torch.nn.functional.normalize(database, dim=1, p=2)
        queries = torch.nn.functional.normalize(queries, dim=1, p=2)

        # similarity
        similarity = torch.mm(queries, database.T)
        distances, indices = similarity.topk(20, largest=True, sorted=True)

        # evaluate
        retrieved = dataset.eval_result(indices, distances)
        print('retrieved', retrieved)
    dist.barrier()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=100, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=16, help='number of workers')
    parser.add_argument('--board_path', '-bp', default='./log', type=str, help='tensorboard path')
    parser.add_argument('--accumulate', default=1, type=int, help='accumulate gradient')
    parser.add_argument('--mlp_hidden', default=4096, type=int, help='mlp hidden dimension')
    parser.add_argument('--ratio', default=1, type=int, help='loss ratio of layer2output')
    parser.add_argument('--up', default=12, type=int, help='layer2high skip layer')
    parser.add_argument('--st_inter', default=False, type=utils.bool_flag, help='intermediate representation of student')
    parser.add_argument('--t_inter', default=False, type=utils.bool_flag, help='intermediate representation of teacher')
    parser.add_argument('--l2o', default=False, type=utils.bool_flag, help='layer2output')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
            distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--ckpt", type=str)

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

    parser.add_argument('--data_path', default='/dataset/copy_detection/', type=str,
                        help="See https://lear.inrialpes.fr/~jegou/data.php#copydays")
    parser.add_argument('--whitening_path', default='/data/dataset/data_yfcc/whitening/', type=str,
                        help="""Path to directory with images used for computing the whitening operator.
            In our paper, we use 20k random images from YFCC100M.""")
    parser.add_argument('--distractors_path', default='/data/dataset/data_yfcc/distractor/', type=str,
                        help="Path to directory with distractors images. In our paper, we use 10k random images from YFCC100M.")

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
    parser.add_argument('--n', default=1, type=int, help="representation layer")

    hparam = parser.parse_args()
    if hparam.load_json:
        with open(hparam.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            hparam = parser.parse_args(namespace=t_args)

    main(hparam)