from utils import *
from master_node import *
from fed_node import *


class AuxCIFAR(nn.Module):
    def __init__(self, n_class, n_channel):
        super(AuxCIFAR, self).__init__()
        self.n_class = n_class
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(n_channel, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, self.n_class),
            nn.Softmax(dim=1)
        )

    def forward(self, X):
        X = self.features(X)
        return self.classifier(X.view(X.size(0), -1))


class SNAS_Architecture(nn.Module):
    def __init__(self, orig_channel, n_channel, genotype, n_base, n_spec, n_ops, n_class, sampler_type='cell_based'):
        super(SNAS_Architecture, self).__init__()
        self.base_arch = MasterNode(orig_channel, n_channel, genotype, n_cell=n_base)
        self.spec_arch = FedNode(n_channel, n_class, genotype, n_cell=n_spec)
        self.base_sampler = SAMPLER[sampler_type](n_base, genotype, n_ops, n_channel)
        self.spec_sampler = SAMPLER[sampler_type](n_spec, genotype, n_ops, n_channel)
        self.aux = AuxCIFAR(n_class, n_channel)

    def forward(self, X, n_sample=5, use_spec=True, use_aux=True):
        X_spec = 0.0
        X_aux = 0.0
        for i in range(n_sample):
            X_base = self.base_arch(X, self.base_sampler, activate=True)
            if use_spec:
                X_spec += self.spec_arch(X_base, self.spec_sampler, activate=True)
            if use_aux:
                X_aux += self.aux(X_base)
        if use_aux:
            X_aux = torch.div(X_aux, n_sample)

        if use_spec:
            X_spec = torch.div(X_spec, n_sample)

        if use_aux:
            if use_spec:
                return X_spec, X_aux
            return X_aux
        elif use_spec:
            return X_spec