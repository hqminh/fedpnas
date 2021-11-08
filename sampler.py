from utils import *

SAMPLER = {
    'sampler': lambda n_cells, genotype, n_ops, n_channel: Sampler(n_cells, genotype, n_ops, n_channel),
    'context_sampler': lambda n_cells, genotype, n_ops, n_channel: ContextSampler(n_cells, genotype, n_ops, n_channel),
    'context_sampler_v2': lambda n_cells, genotype, n_ops, n_channel: ContextSamplerv2(n_cells, genotype, n_ops, n_channel),
    'cell_based': lambda n_cells, genotype, n_ops, n_channel: CellBasedContextSampler(n_cells, genotype, n_ops, n_channel),
    'cell_based_v2': lambda n_cells, genotype, n_ops, n_channel: CellBasedContextSamplerv2(n_cells, genotype, n_ops, n_channel)
}


class Sampler(nn.Module):
    def __init__(self, n_cells, genotype, n_ops, n_channel, temperature=0.1):
        super(Sampler, self).__init__()
        self.n_cells, self.n_ops, self.temperature, self.n_channel = n_cells, n_ops, temperature, n_channel
        self.genotype = genotype
        self.alpha_net = nn.ModuleList([
            nn.Linear(1, self.genotype.n_edge * self.n_ops)
            for _ in range(self.n_cells)
        ]).to(device)

    def _sample(self, alpha):
        return F.gumbel_softmax(alpha, self.temperature, hard=True)

    def sample(self, X, cell_id, n_sample=1):
        # Does not use X  but still pass in anyway to simplify code
        one = torch.ones(1, 1).to(device)
        alpha = F.tanh(self.alpha_net[cell_id](one).view(1, self.genotype.n_edge, self.n_ops))
        return torch.stack([self._sample(alpha) for _ in range(n_sample)])


class CellBasedContextSampler(nn.Module):
    def __init__(self, n_cells, genotype, n_ops, n_channel, temperature=0.1):
        super(CellBasedContextSampler, self).__init__()
        self.n_cells, self.n_ops, self.temperature = n_cells, n_ops, temperature
        self.genotype = genotype
        # Output is n_sample x 1 x H * W
        self.conv_layer = nn.ModuleList([
            nn.Conv2d(n_channel, 1, kernel_size=1, stride=1, padding=0)
            for _ in range(n_cells)
        ]).to(device)
        # Output is n_sample x 1 x n_edge x n_ops
        self.global_pool = nn.AdaptiveAvgPool2d((genotype.n_edge, n_ops)).to(device)
        self.op_list = list(SMALL_OPS.keys())

    def _sample(self, alpha):
        return F.gumbel_softmax(alpha, self.temperature, hard=True)

    def sample(self, X, cell_id, n_sample=1):
        alpha = torch.mean(F.tanh(self.global_pool(F.relu(self.conv_layer[cell_id](X)))), dim=0)
        return torch.stack([self._sample(alpha) for _ in range(n_sample)])

    def visualize_cell_realization(self, Z):
        Z = torch.argmax(Z, dim=-1)
        for cell in range(self.n_cells):
            print(f'Cell {cell}')
            for edge in self.genotype.edge_list.keys():
                edge_idx = self.genotype.edge_list[edge]
                print(f'    Edge: {edge}, ops: {self.op_list[Z[cell][edge_idx]]}')


class ContextSampler(nn.Module):
    def __init__(self, n_cells, genotype, n_ops, n_channel, temperature=0.1):
        super(ContextSampler, self).__init__()
        self.n_cells, self.n_ops, self.temperature = n_cells, n_ops, temperature
        self.genotype = genotype
        self.conv_layer = nn.Conv2d(n_channel, n_cells, kernel_size=3, stride=1, padding=1).to(device)
        self.global_pool = nn.AdaptiveAvgPool2d((genotype.n_edge, n_ops)).to(device)
        self.op_list = list(FULL_OPS.keys())

    def _sample(self, alpha):
        return F.gumbel_softmax(alpha, self.temperature, hard=True)
        #self.visualize_cell_realization(Z)

    def sample(self, X, n_sample=100):
        alpha = torch.mean(F.tanh(self.global_pool(F.relu(self.conv_layer(X)))), dim=0)
        return torch.stack([self._sample(alpha) for _ in range(n_sample)])

    def visualize_cell_realization(self, Z):
        Z = torch.argmax(Z, dim=-1)
        for cell in range(self.n_cells):
            print(f'Cell {cell}')
            for edge in self.genotype.edge_list.keys():
                edge_idx = self.genotype.edge_list[edge]
                print(f'    Edge: {edge}, ops: {self.op_list[Z[cell][edge_idx]]}')


class ContextSamplerv2(nn.Module):
    def __init__(self, n_cells, genotype, n_ops, n_channel, temperature=0.1):
        super(ContextSamplerv2, self).__init__()
        self.n_cells, self.n_ops, self.temperature = n_cells, n_ops, temperature
        self.genotype = genotype
        self.conv_layer = nn.Conv2d(n_channel, n_cells, kernel_size=3, stride=1, padding=1).to(device)
        self.global_pool = nn.AdaptiveAvgPool2d((genotype.n_edge, n_ops)).to(device)
        self.op_list = list(SMALL_OPS.keys())

    def _sample(self, alpha):
        return F.gumbel_softmax(alpha, self.temperature, hard=True).view(self.n_cells, self.genotype.n_edge, self.n_ops)
        #self.visualize_cell_realization(Z)

    def sample(self, X, n_sample=100):
        alpha = torch.mean(F.tanh(self.global_pool(F.relu(self.conv_layer(X)))).view(X.shape[0], self.n_cells * self.genotype.n_edge, self.n_ops), dim = 0)
        return torch.stack([self._sample(alpha) for _ in range(n_sample)])

    def visualize_cell_realization(self, Z):
        Z = torch.argmax(Z, dim=-1)
        for cell in range(self.n_cells):
            print(f'Cell {cell}')
            for edge in self.genotype.edge_list.keys():
                edge_idx = self.genotype.edge_list[edge]
                print(f'    Edge: {edge}, ops: {self.op_list[Z[cell][edge_idx]]}')


class CellBasedContextSamplerv2(nn.Module):
    def __init__(self, n_cells, genotype, n_ops, n_channel, temperature=0.1):
        super(CellBasedContextSamplerv2, self).__init__()
        self.n_cells, self.n_ops, self.temperature = n_cells, n_ops, temperature
        self.genotype = genotype
        self.conv_layer = nn.ModuleList([
            nn.Conv2d(n_channel, 1, kernel_size=1, stride=1, padding=0)
            for _ in range(n_cells)
        ]).to(device)
        self.global_pool = nn.AdaptiveAvgPool2d((genotype.n_edge, n_ops)).to(device)
        self.op_list = list(SMALL_OPS.keys())

    def _sample(self, alpha):
        return F.gumbel_softmax(alpha, self.temperature, hard=True).view(1, self.genotype.n_edge, self.n_ops)

    def sample(self, X, cell_id, n_sample=1):
        alpha = torch.mean(
            F.tanh(
                self.global_pool(F.relu(self.conv_layer[cell_id](X)))
            ).view(X.shape[0], self.genotype.n_edge, self.n_ops)
            , dim=0
        )
        return torch.stack([self._sample(alpha) for _ in range(n_sample)])

    def visualize_cell_realization(self, Z):
        Z = torch.argmax(Z, dim=-1)
        for cell in range(self.n_cells):
            print(f'Cell {cell}')
            for edge in self.genotype.edge_list.keys():
                edge_idx = self.genotype.edge_list[edge]
                print(f'    Edge: {edge}, ops: {self.op_list[Z[cell][edge_idx]]}')