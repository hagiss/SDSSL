from torch import nn
import torch
from einops import rearrange, repeat


# MLP class for projector and predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    def __init__(self, net, embed_size, args, prediction=False, intermediate=False):
        super().__init__()
        self.net = net
        self.intermediate = intermediate
        self.predictor = None
        self.projection_hidden_size = args.mlp_hidden

        self.up = args.up

        if intermediate is False:
            self.projector = MLP(embed_size, args.out_dim, args.mlp_hidden)
            if prediction:
                self.predictor = MLP(args.out_dim, args.out_dim, args.mlp_hidden)

        else:
            self.projector = nn.ModuleList([])

            if prediction:
                self.predictor = nn.ModuleList([])

            for i in range(12):
                mlp = MLP(embed_size, args.out_dim, args.mlp_hidden)
                self.projector.append(mlp)

                if prediction:
                    mlp2 = MLP(args.out_dim, args.out_dim, args.mlp_hidden)
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
                representation = rearrange(representation, "(d b) e -> d b e", d=12)
                for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
                    ret.append(predict(project(representation[i, :])))
                ret = torch.cat(ret)
                # shape: [(d b) e] -> [12*batch, e]
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
                ret = self.predictor(ret)

        return ret
