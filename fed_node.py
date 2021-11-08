from utils import *
from cell import *


class FedNode(nn.Module):
    def __init__(self, n_channel, n_class, genotype, n_cell=N_SPEC_CELL):
        super(FedNode, self).__init__()
        self.n_channel, self.genotype, self.n_cell, self.n_class = n_channel, genotype, n_cell, n_class
        self.reduce_list = [n_cell - 1]
        self.spec_stack = nn.ModuleList([])
        for i in range(self.n_cell):
            self.spec_stack.append(Cell(genotype, n_channel=self.n_channel, reduction=(i in self.reduce_list)))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.n_channel, self.n_class),
            nn.Softmax(dim=1)
        )

    def activate(self, Z):
        for i, cell in enumerate(self.spec_stack):
            cell.activate(Z[i])

    def forward(self, X, sampler, activate=True):
        X0, X1 = X, X
        for i in range(self.n_cell):
            Zi = torch.squeeze(sampler.sample(torch.cat([X0, X1]), cell_id=i, n_sample=1))
            if activate:
                self.spec_stack[i].activate(Zi)
            X2 = self.spec_stack[i](Zi, X0, X1)
            if self.spec_stack[i].reduction:
                X0, X1 = X2, X2
            else:
                X0, X1 = X1, X2
            del Zi
            torch.cuda.empty_cache()
        res = self.global_pooling(X1)  # n_batch by n_channel by 1 by 1
        return self.classifier(res.view(res.size(0), -1))