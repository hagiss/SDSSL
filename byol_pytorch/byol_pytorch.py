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
        self.mlp_layers = 12

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

            for i in range(self.mlp_layers):
                if i == self.mlp_layers-1:
                    mlp = MLP(3, embed_size, args.out_dim, args.mlp_hidden, last_bn)
                else:
                    mlp = MLP(3, embed_size, args.out_dim, int(args.mlp_hidden/2), last_bn)

                self.projector.append(mlp)

                if prediction:
                    if i == self.mlp_layers-1:
                        mlp2 = MLP(2, args.out_dim, args.out_dim, args.mlp_hidden, last_bn)
                    else:
                        mlp2 = MLP(2, args.out_dim, args.out_dim, args.mlp_hidden, last_bn)
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
            representation = rearrange(representation, "(d b) e -> d b e", d=self.mlp_layers)
            for i, project in enumerate(self.projector):
                ret.append(project(representation[i, :]))

            # s0 = torch.cuda.Stream()
            # s1 = torch.cuda.Stream()
            # s2 = torch.cuda.Stream()
            # s3 = torch.cuda.Stream()
            # s4 = torch.cuda.Stream()
            # s5 = torch.cuda.Stream()
            # s6 = torch.cuda.Stream()
            # s7 = torch.cuda.Stream()
            # s8 = torch.cuda.Stream()
            # s9 = torch.cuda.Stream()
            # s10 = torch.cuda.Stream()
            # s11 = torch.cuda.Stream()
            #
            # with torch.cuda.stream(s0):
            #     rep0 = self.projector[0](representation[0, :])
            # with torch.cuda.stream(s1):
            #     rep1 = self.projector[1](representation[1, :])
            # with torch.cuda.stream(s2):
            #     rep2 = self.projector[2](representation[2, :])
            # with torch.cuda.stream(s3):
            #     rep3 = self.projector[3](representation[3, :])
            # with torch.cuda.stream(s4):
            #     rep4 = self.projector[4](representation[4, :])
            # with torch.cuda.stream(s5):
            #     rep5 = self.projector[5](representation[5, :])
            # with torch.cuda.stream(s6):
            #     rep6 = self.projector[6](representation[6, :])
            # with torch.cuda.stream(s7):
            #     rep7 = self.projector[7](representation[7, :])
            # with torch.cuda.stream(s8):
            #     rep8 = self.projector[8](representation[8, :])
            # with torch.cuda.stream(s9):
            #     rep9 = self.projector[9](representation[9, :])
            # with torch.cuda.stream(s10):
            #     rep10 = self.projector[10](representation[10, :])
            # with torch.cuda.stream(s11):
            #     rep11 = self.projector[11](representation[11, :])

            # torch.cuda.synchronize()

            # ret = torch.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8, rep9, rep10, rep11])
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

        return ret

    # x.shape is [12 * batch, out_dim]
    def predict(self, x, d=12):
        if self.predictor is None:
            return x

        if self.intermediate and d > 1:
            ret = []
            projections = rearrange(x, "(d b) e -> d b e", d=d)
            for i, predictor in enumerate(self.predictor):
                if i == d:
                    break
                prediction = predictor(projections[i, :])
                ret.append(prediction)

            # s0 = torch.cuda.Stream()
            # s1 = torch.cuda.Stream()
            # s2 = torch.cuda.Stream()
            # s3 = torch.cuda.Stream()
            # s4 = torch.cuda.Stream()
            # s5 = torch.cuda.Stream()
            # s6 = torch.cuda.Stream()
            # s7 = torch.cuda.Stream()
            # s8 = torch.cuda.Stream()
            # s9 = torch.cuda.Stream()
            # s10 = torch.cuda.Stream()
            # s11 = torch.cuda.Stream()
            #
            # with torch.cuda.stream(s0):
            #     rep0 = self.predictor[0](projections[0, :])
            # with torch.cuda.stream(s1):
            #     rep1 = self.predictor[1](projections[1, :])
            # with torch.cuda.stream(s2):
            #     rep2 = self.predictor[2](projections[2, :])
            # with torch.cuda.stream(s3):
            #     rep3 = self.predictor[3](projections[3, :])
            # with torch.cuda.stream(s4):
            #     rep4 = self.predictor[4](projections[4, :])
            # with torch.cuda.stream(s5):
            #     rep5 = self.predictor[5](projections[5, :])
            # with torch.cuda.stream(s6):
            #     rep6 = self.predictor[6](projections[6, :])
            # with torch.cuda.stream(s7):
            #     rep7 = self.predictor[7](projections[7, :])
            # with torch.cuda.stream(s8):
            #     rep8 = self.predictor[8](projections[8, :])
            # with torch.cuda.stream(s9):
            #     rep9 = self.predictor[9](projections[9, :])
            # with torch.cuda.stream(s10):
            #     rep10 = self.predictor[10](projections[10, :])
            # with torch.cuda.stream(s11):
            #     rep11 = self.predictor[11](projections[11, :])
            #
            # ret = torch.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8, rep9, rep10, rep11])

            return torch.cat(ret)
        elif self.intermediate:
            return self.predictor[-1](x)
        return self.predictor(x)


    # def forward(self, x, return_embedding=False, epoch=None):
    #     # if self.predictor is not None and return_embedding is False:
    #     #     representation = self.net.get_intermediate_layers(x, 12)
    #     # else:
    #     #     representation = self.net(x)
    #     if self.intermediate and return_embedding is False:
    #         representation = self.net.get_intermediate_layers(x, 12)
    #     else:
    #         representation = self.net(x)
    #
    #     if return_embedding:
    #         return representation
    #
    #     if self.intermediate:
    #         ret = []
    #         if self.predictor is not None:
    #             ret_pred = []
    #             ret_proj = []
    #             representation = rearrange(representation, "(d b) e -> d b e", d=12)
    #             # for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
    #             for i, (project, predict) in enumerate(zip(self.projector, self.predictor)):
    #                 proj = project(representation[i, :])
    #                 ret_proj.append(proj)
    #                 # proj_detached = project(representation[i, :].detach())
    #                 ret.append(predict(proj))
    #                 ret_pred.append(predict(proj.detach()))
    #             ret = torch.cat(ret)
    #             ret_pred = torch.cat(ret_pred)
    #             ret_proj = torch.cat(ret_proj)
    #             # shape: [(d b) e] -> [12*batch, e]
    #             return ret, ret_pred, ret_proj
    #         else:
    #             representation = rearrange(representation, "(d b) e -> d b e", d=12)
    #             for i, project in enumerate(self.projector):
    #                 ret.append(project(representation[i, :]))
    #
    #             if self.up > 0:
    #                 last = ret[-1].unsqueeze(0)
    #                 last = repeat(last, "() b e -> (d b) e", d=self.up)
    #                 ret = torch.cat(ret[self.up:])
    #                 ret = torch.cat([ret, last])
    #             else:
    #                 ret = torch.cat(ret)
    #             # shape: [(d b) e] -> [12*batch, e]
    #     else:
    #         ret = self.projector(representation)
    #         if self.predictor is not None:
    #             ret_pred = self.predictor(ret.detach())
    #             ret = self.predictor(ret)
    #             return ret, ret_pred
    #
    #     return ret
