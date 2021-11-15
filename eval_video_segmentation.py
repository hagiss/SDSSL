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

import utils
import json
import numpy as np
import vision_transformer as vits

from PIL import Image
from pl_train_moco import PLLearner


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

n_last_frames = 7
@torch.no_grad()
def eval_video_tracking_davis(model, frame_list, video_dir, first_seg, seg_ori, color_palette, output_dir, n=1):
    """
    Evaluate tracking on a video given first frame & segmentation
    """
    video_folder = os.path.join(output_dir, video_dir.split('/')[-1])
    os.makedirs(video_folder, exist_ok=True)

    # The queue stores the n preceeding frames
    que = queue.Queue(n_last_frames)

    # first frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # extract first frame feature
    frame1_feat = extract_feature(model, frame1, n=n).T  # dim x h*w

    # saving first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    mask_neighborhood = None
    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = read_frame(frame_list[cnt])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(model, frame_tar, used_frame_feats,
                                                                       used_segs, n, mask_neighborhood)

        # pop out oldest frame if neccessary
        if que.qsize() == n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=32, mode='bilinear', align_corners=False,
                                      recompute_scale_factor=False)[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split('/')[-1].replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


size_mask_neighborhood = 12


def restrict_neighborhood(h, w):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if (mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def label_propagation(model, frame_tar, list_frame_feats, list_segs, n, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    ## we only need to extract feature of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, n=n, return_h_w=True)

    return_feat_tar = feat_tar.T  # dim x h*w

    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w

    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)  # nmb_context x h*w (tar: query) x h*w (source: keys)

    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.transpose(2, 1).reshape(-1, h * w)  # nmb_context*h*w (source: keys) x h*w (tar: queries)
    tk_val, _ = torch.topk(aff, dim=0, k=5)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T  # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    return seg_tar, return_feat_tar, mask_neighborhood


def extract_feature(model, frame, n=1, return_h_w=False):
    """Extract one frame feature everytime."""
    out = model.get_intermediate_layers_all(frame.unsqueeze(0).cuda(), n=n)[0].clone()
    out = out[:, 1:, :]  # we discard the [CLS] token
    h, w = int(frame.shape[1] / model.patch_embed.patch_size[0]), int(frame.shape[2] / model.patch_embed.patch_size[1])
    dim = out.shape[-1]
    out = out[0].reshape(h, w, dim)
    out = out.reshape(-1, dim)
    if return_h_w:
        return out, h, w
    return out


def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if (n_dims is None):
        n_dims = int(y_tensor.max() + 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def read_frame_list(video_dir):
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir, scale_size=[480]):
    """
    read a single frame & preprocess
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if (ori_h > ori_w):
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def read_seg(seg_dir, factor, scale_size=[480]):
    seg = Image.open(seg_dir)
    _w, _h = seg.size  # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if (_w > _h):
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return to_one_hot(small_seg), np.asarray(seg)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def main(args):

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
    #
    # # student = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)
    # # teacher = torchvision_models.resnet18(pretrained=False, num_classes=args.out_dim)

    lr = args.lr * 10000
    min_lr = args.min_lr * 10000
    total_batch = torch.cuda.device_count() * args.batch_size_per_gpu
    clip = args.clip_grad

    args.image_size = 224
    args.total_batch = total_batch
    args.optimizer = 'adamw'
    args.temperature = 0.2
    # args.st_inter = False

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


    # ============ Extract features ... ============
    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

    data_path = "/dataset/davis-2017/DAVIS/"

    video_list = open(os.path.join(data_path, "ImageSets/2017/val.txt")).readlines()
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        print(f'[{i}/{len(video_list)}] Begin to segmentate video {video_name}.')
        video_dir = os.path.join(data_path, "JPEGImages/480p/", video_name)
        frame_list = read_frame_list(video_dir)
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        first_seg, seg_ori = read_seg(seg_path, 32)
        eval_video_tracking_davis(model, frame_list, video_dir, first_seg, seg_ori, color_palette, args.output_dir, args.n)


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
    parser.add_argument('--output_dir', type=str, help="output_dir")
    parser.add_argument('--ckpt', type=str, help="checkpoint")
    parser.add_argument('--n', type=int, help='layer')

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