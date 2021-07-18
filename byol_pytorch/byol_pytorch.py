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
    def __init__(self, net, embed_size, args, prediction=False):
        super().__init__()
        self.net = net
        self.predictor = None

        self.projection_hidden_size = args.mlp_hidden
        # self.projector = MLP(embed_size, args.out_dim, args.mlp_hidden)

        self.projector = None
        if prediction:
            # self.predictor = MLP(args.out_dim, args.out_dim, args.mlp_hidden)
            self.projector = nn.ModuleList([])
            self.predictor = nn.ModuleList([])
            for i in range(12):
                hidden = args.mlp_hidden if i is 11 else int(args.mlp_hidden/args.div)
                mlp = MLP(embed_size, args.out_dim, hidden)
                mlp2 = MLP(args.out_dim, args.out_dim, args.mlp_hidden)
                self.projector.append(mlp)
                self.predictor.append(mlp2)
        else:
            self.projector = MLP(embed_size, args.out_dim, args.mlp_hidden)

    def get_representation(self, x):
        return self.forward(x, True)

    def forward(self, x, return_embedding=False):
        if self.predictor is not None and return_embedding is False:
            representation = self.net.get_intermediate_layers(x, 12)
        else:
            representation = self.net(x)

        if return_embedding:
            return representation

        ret = []
        if self.predictor is not None:
            representation = rearrange(representation, "(d b) e -> d b e", d=12)
            for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
                ret.append(predict(project(representation[i, :])))
            ret = torch.cat(ret)
            # shape: [(d b) e] -> [12*batch, e]
        else:
            ret = self.projector(representation).unsqueeze(0)
            ret = repeat(ret, "() b e -> (d b) e", d=12)
            # shape: [(d b) e] -> [12*batch, e]

        # ret = self.projector(representation)
        # if self.predictor is not None:
        #     ret = self.predictor(ret)

        return ret
