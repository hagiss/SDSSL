from torch import nn
import torch
from einops import rearrange, repeat


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, num_layers, dim, projection_size, hidden_size=4096, last_bn=True):
        super().__init__()
        mlp = []
        for l in range(num_layers):
            dim1 = dim if l == 0 else hidden_size
            dim2 = projection_size if l == num_layers - 1 else hidden_size

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        return self.net(x)

class MLP_wo_batch(nn.Module):
    def __init__(self, num_layers, dim, projection_size, hidden_size=4096, last_bn=False):
        super().__init__()
        mlp = []
        for l in range(num_layers):
            dim1 = dim if l == 0 else hidden_size
            dim2 = projection_size if l == num_layers - 1 else hidden_size

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                # mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, embed_size, args, prediction=False, intermediate=False, last_bn=True):
        super().__init__()
        self.net = net
        self.intermediate = intermediate
        self.predictor = None
        self.projection_hidden_size = args.mlp_hidden

        # if prediction:
        #     self.predictor = MLP(args.out_dim, args.out_dim, args.mlp_hidden)
        #     self.dummy_predictor = MLP(args.out_dim, args.out_dim, args.mlp_hidden)


        self.up = args.up

        if intermediate is False:
            self.projector = MLP(3, embed_size, args.out_dim, args.mlp_hidden, last_bn)
            if prediction:
                self.predictor = MLP(2, args.out_dim, args.out_dim, args.mlp_hidden, last_bn)

        else:
            self.projector = nn.ModuleList([])

            if prediction:
                self.predictor = nn.ModuleList([])

            for i in range(12):
                if i == 11:
                    mlp = MLP(3, embed_size, args.out_dim, args.mlp_hidden, last_bn)
                else:
                    mlp = MLP_wo_batch(2, embed_size, args.out_dim, args.mlp_hidden, last_bn)

                self.projector.append(mlp)

                if prediction:
                    if i == 11:
                        mlp2 = MLP(2, args.out_dim, args.out_dim, args.mlp_hidden, last_bn)
                    else:
                        mlp2 = MLP_wo_batch(2, args.out_dim, args.out_dim, args.mlp_hidden, last_bn)
                    self.predictor.append(mlp2)

    def get_representation(self, x):
        return self.forward(x, True)

    def forward(self, x, return_embedding=False, epoch=None):
        # if self.predictor is not None and return_embedding is False:
        #     representation = self.net.get_intermediate_layers(x, 12)
        # else:
        #     representation = self.net(x)
        if self.intermediate and return_embedding is False:
            representation = self.net.get_intermediate_layers(x, 12)
        else:
            representation = self.net(x)

        if return_embedding:
            return representation

        if self.intermediate:
            ret = []
            if self.predictor is not None:
                ret_pred = []
                representation = rearrange(representation, "(d b) e -> d b e", d=12)
                # for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
                for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
                    proj = project(representation[i, :])
                    ret.append(predict(proj))
                    ret_pred.append(predict(proj.detach()))
                ret = torch.cat(ret)
                ret_pred = torch.cat(ret_pred)
                # shape: [(d b) e] -> [12*batch, e]
                return ret, ret_pred
            else:
                representation = rearrange(representation, "(d b) e -> d b e", d=12)
                for i, project in enumerate(self.projector):
                    ret.append(project(representation[i, :]))

                if self.up > 0:
                    last = ret[-1].unsqueeze(0)
                    last = repeat(last, "() b e -> (d b) e", d=self.up)
                    ret = torch.cat(ret[self.up:])
                    ret = torch.cat([ret, last])
                else:
                    ret = torch.cat(ret)
                # shape: [(d b) e] -> [12*batch, e]
        else:
            ret = self.projector(representation)
            if self.predictor is not None:
                ret_pred = self.predictor(ret.detach())
                ret = self.predictor(ret)
                return ret, ret_pred

        return ret
