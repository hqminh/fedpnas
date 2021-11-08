from functools import lru_cache
from utils import *
from genotype import *


class Cell(nn.Module):
    def __init__(self, genotype, n_channel=1, reduction=False):  # graph is n_node by n_node by n_ops
        super(Cell, self).__init__()
        self.C, self.reduction = n_channel, reduction
        self.ops = [SMALL_OPS[key] for key in SMALL_OPS.keys()]
        self.n_ops = len(self.ops)
        self.model = nn.ModuleList([])  # all learnable parameters
        self.genotype = genotype
        for (u, v) in self.genotype.edge_list.keys():
            edge = nn.ModuleList([])
            st = 2 if (u in genotype.input and self.reduction) else 1
            for i in range(self.n_ops):
                edge.append(self.ops[i](self.C, st, True))
            self.model.append(edge)

    def activate(self, Z):
        for edge in range(self.genotype.n_edge):
            for o in range(self.n_ops):
                if Z[edge, o] >= 0.5:
                    unfreeze(self.model[edge][o])

    def forward(self, Z, X0, X1):
        @lru_cache(maxsize=None)
        def _compute(v):
            if v == 0: return X0
            if v == 1: return X1
            res = []
            for u in self.genotype.adj_list[v]:
                edge = self.genotype.edge_list[(u, v)]
                ops_max = torch.argmax(Z[edge])
                temp = self.model[edge][ops_max](_compute(u))
                res.append(Z[edge, ops_max] * temp)
            return torch.mean(torch.stack(res), dim=0)

        values = []
        for tv in self.genotype.terminal:
            values.append(_compute(tv))
        return torch.mean(torch.stack(values), dim=0)
