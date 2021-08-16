from byol import train
import json
import argparse
from torchvision import models as torchvision_models
from torchvision import transforms as T
from PIL import Image


torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))

def run(num_gpus=1):
    parser = argparse.ArgumentParser(description='byol')

    parser.add_argument('--load_json',
                        help='Load settings from file in json format. Command line options override values in file.')

    parser.add_argument('--lr', '-l', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=300, help="epochs for scheduling")
    parser.add_argument('--max_epochs', type=int, default=100, help="epochs for actual training")
    parser.add_argument('--batch_size_per_gpu', '-b', type=int, default=256, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=16, help='number of workers')
    parser.add_argument('--board_path', '-bp', default='./log', type=str, help='tensorboard path')
    parser.add_argument('--mlp_hidden', default=4096, type=int, help='mlp hidden dimension')
    parser.add_argument('--ratio', default=1, type=int, help='loss ratio of layer2output')
    parser.add_argument('--up', default=12, type=int, help='layer2high skip layer')
    parser.add_argument('--st_inter', default=False, type=bool, help='intermediate representation of student')
    parser.add_argument('--t_inter', default=False, type=bool, help='intermediate representation of teacher')

    parser.add_argument('--data', '-d', metavar='DIR', default='../dataset',
                        help='path to dataset')
    parser.add_argument('--dataset', '-ds', default='stl10',
                        help='dataset name', choices=['stl10', 'cifar10', 'imagenet'])
    parser.add_argument('--name', help='name for tensorboard')
    parser.add_argument('--accelerator', default='ddp', type=str,
                        help='ddp for multi-gpu or node, ddp2 for across negative samples')

    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
                end of optimization. We use a cosine LR schedule with linear warmup.""")
    # parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
    #     during which we keep the output layer fixed. Typically doing so during
    #     the first epoch helps training. Try increasing this value if the loss does not decrease.""")
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
    parser.add_argument('--div', default=4, type=int, help="dividing hidden dimensions of mlp1")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
                parameter for teacher update. The value is increased to 1 during training with cosine schedule.
                We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    hparam = parser.parse_args()
    if hparam.load_json:
        with open(hparam.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            hparam = parser.parse_args(namespace=t_args)

    pretrain_transform = T.Compose([
        T.Resize((256, 256), interpolation=Image.BICUBIC),
        T.ToTensor(),
    ])
    # path = '/data/dataset/imagenet_cls_loc/CLS_LOC/ILSVRC2015/Data/CLS-LOC'
    # dataset = datasets.ImageFolder(
    #     path + '/train',
    #     pretrain_transform
    # )

    try:
        # TODO define dataset
        dataset = None

        train(dataset, hparam)

    except Exception as e:
        print(e)
