from utils import *
from cell import *
from sampler import *

'''
CENTRAL PHASE:
Local queries base embedding from central
Central returns base embedding to local
Local computes prediction loss given embedding
Local sends loss to Central
Central computes average loss and update base cells

LOCAL PHASE:
Local queries base embedding from central
Central returns base embedding to local
Local computes prediction loss given embedding
Local updates local cells
'''


class MasterNode(nn.Module):
    def __init__(self, orig_channel, n_channel, genotype, n_cell=N_BASE_CELL):
        super(MasterNode, self).__init__()
        self.orig_channel, self.n_channel, self.genotype, self.n_cell = orig_channel, n_channel, genotype, n_cell
        self.embedding = nn.Conv2d(orig_channel, n_channel, kernel_size=5, stride=1, padding=2)
        self.reduce_list = [self.n_cell // 3, 2 * self.n_cell // 3]
        self.base_stack = nn.ModuleList([])
        for i in range(self.n_cell):
            self.base_stack.append(Cell(genotype, n_channel=self.n_channel, reduction=(i in self.reduce_list)))

    def activate(self, Z):
        for i, cell in enumerate(self.base_stack):
            cell.activate(Z[i])

    def forward(self, X, sampler, activate=True):
        X = self.embedding(X)
        X0, X1 = X, X
        for i in range(self.n_cell):
            Zi = torch.squeeze(sampler.sample(torch.cat([X0, X1]), cell_id=i, n_sample=1))
            if activate:
                self.base_stack[i].activate(Zi)
            X2 = self.base_stack[i](Zi, X0, X1)
            if self.base_stack[i].reduction:
                X0, X1 = X2, X2
            else:
                X0, X1 = X1, X2

            del Zi
            torch.cuda.empty_cache()
        return X1