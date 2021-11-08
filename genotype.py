from utils import *

class CellGenotype:
    def __init__(self, n_source=2, n_sink=1, n_hidden=2, max_skip=None):
        self.n_source, self.n_sink, self.n_hidden = n_source, n_sink, n_hidden
        self.n_node = n_source + n_sink + n_hidden
        self.terminal = [self.n_node - i - 1 for i in range(n_sink)]
        self.input = [i for i in range(n_source)]
        self.max_skip = max_skip if max_skip is not None else self.n_node - 1
        self.n_edge, self.edge_list, self.adj_list = self.generate_edge_list()

    def generate_edge_list(self):
        n_edge = 0
        edge_list = {}
        adj_list = {}
        for v in range(self.n_source, self.n_node):
            adj_list[v] = []
            for u in range(v):
                if u in self.terminal:
                    continue #terminal node has no outgoing edge
                # if target v in terminal, then v is counted as the smallest index terminal
                # if source u in source, then u is counted as the largest index source
                skip = min(self.n_node - self.n_sink, v) - max(self.n_source - 1, u)
                if skip <= self.max_skip:
                    edge_list[(u, v)] = n_edge
                    n_edge += 1
                    adj_list[v].append(u)

        return n_edge, edge_list, adj_list


class BasicSeqGenotype:
    def __init__(self, C_in):
        self.genotypes = []
        self.C = [C_in]
        self.reduction = []

    def add_cell(self, c, genotype, reduce=False):
        self.C.append(c)
        self.genotypes.append(genotype)
        self.reduction.append(reduce)

    def query_cell(self, i):
        assert i < len(self.genotypes)
        assert i >= 0
        return self.C[i], self.C[i + 1], self.reduction, self.genotypes[i]


# class MasterGenotype_v1(BasicSeqGenotype):
#     def __init__(self, C_in):
#         super(MasterGenotype_v1, self).__init__(C_in)
#         self.add_cell(8, CellGenotype(2, 3, 2, max_skip=2, op_list=SMALL_OPS))
#         self.add_cell(8, CellGenotype(2, 2, 1, max_skip=1, op_list=LARGE_OPS), reduce=True)
#         self.add_cell(16, CellGenotype(2, 3, 1, max_skip=2, op_list=SMALL_OPS))
#         self.add_cell(16, CellGenotype(2, 3, 1, max_skip=2, op_list=SMALL_OPS))
#         self.add_cell(32, CellGenotype(2, 2, 1, max_skip=1, op_list=LARGE_OPS), reduce=True)
#         self.add_cell(64, CellGenotype(2, 1, 1, max_skip=1, op_list=FULL_OPS))
#
#
# class NodeGenotype_v1(BasicSeqGenotype):
#     def __init__(self, C_in):
#         super(NodeGenotype_v1, self).__init__(C_in)
#         self.add_cell(128, CellGenotype(2, 2, 2, max_skip=1, op_list=SMALL_OPS))
#         self.add_cell(768, CellGenotype(2, 0, 1, max_skip=1, op_list=CONVS), reduce=True)


if __name__ == '__main__':
    pass