from utils import *
from architecture import *


# Implement FedPNAS server
class Server:
    def __init__(self, architecture):
        self.loss = nn.NLLLoss()
        self.architecture = architecture
        self.architecture.to(device)

    def fed_average(self, client_architectures):
        state_dicts = []
        for c_arch in client_architectures:
            state_dicts.append(c_arch)

        server_state_dict = deepcopy(state_dicts[0])
        for k in server_state_dict.keys():
            for i in range(1, len(state_dicts)):
                server_state_dict[k] += state_dicts[i][k]
            server_state_dict[k] = torch.div(server_state_dict[k], len(client_architectures))

        self.architecture.load_state_dict(server_state_dict)