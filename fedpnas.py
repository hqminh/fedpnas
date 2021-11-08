from cifar_utils import *
from mnist_utils import *
from client import *
from server import *


class FedPNAS:
    def __init__(self, orig_channel, n_channel, n_client, n_base, n_spec, n_ops, n_class, genotype, gpu_device,
                 Xs, Ys, Xt, Yt, sampler_type='cell_based', save_folder='./model_chkpoint', meta_task=True):
        self.save_folder = save_folder
        self.meta_task = meta_task
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        architecture = SNAS_Architecture(
            orig_channel, n_channel, genotype, n_base, n_spec,
            n_ops, n_class, sampler_type=sampler_type
        )
        self.device = gpu_device
        self.server = Server(architecture)
        self.clients = []
        for i in range(n_client):
            self.clients.append(Client(deepcopy(architecture), client_id=i))
            self.clients[i].load_data(Xs[i], Ys[i])

        if self.meta_task:
            self.meta_test_dataset = [
                DataLoader(TensorDataset(Xt[i], Yt[i]), batch_size=BATCH_SIZE, shuffle=True)
                for i in range(n_client)
            ]
            self.test_dataset = None
        else:
            self.meta_test_dataset = None
            self.test_dataset = DataLoader(TensorDataset(Xt, Yt), batch_size=BATCH_SIZE, shuffle=True)

    def eval_meta(self, n_sample=5):
        acc = np.zeros(len(self.clients))
        with torch.no_grad():
            for client_id, client in enumerate(self.clients):
                n_test = 0
                tqdm_loader = tqdm(self.meta_test_dataset[client_id], desc=f'Evaluate Client {client_id}', leave=True)
                for batch, (X, Y) in enumerate(tqdm_loader):
                    X, Y = X.to(device), Y.to(device)
                    X_spec = client.architecture(X, n_sample, use_spec=True, use_aux=False)
                    batch_acc = compute_accuracy(X_spec, Y).item()
                    acc[client_id] += batch_acc * X.shape[0]
                    n_test += X.shape[0]
                    tqdm_loader.set_postfix_str(f'Batch accuracy = {batch_acc}, Total acc= {acc[client_id]}')
                acc[client_id] /= n_test

        return acc

    def eval_aux_meta(self, n_sample=5):
        acc = np.zeros(len(self.clients))
        with torch.no_grad():
            for client_id, client in enumerate(self.clients):
                n_test = 0.0
                tqdm_loader = tqdm(self.meta_test_dataset[client_id], desc=f'Evaluate Client {client_id}', leave=True)
                for batch, (X, Y) in enumerate(tqdm_loader):
                    X, Y = X.to(device), Y.to(device)
                    X_aux = client.architecture(X, n_sample, use_spec=False, use_aux=True)
                    batch_acc = compute_accuracy(X_aux, Y).item()
                    acc[client_id] += batch_acc * X.shape[0]
                    n_test += X.shape[0]
                    tqdm_loader.set_postfix_str(f'Batch accuracy = {batch_acc:.3f}, Total acc= {acc[client_id]:.3f}')
                acc[client_id] /= n_test
        return acc

    def eval(self, n_sample=5):
        acc = np.zeros(len(self.clients))
        n_test = 0
        tqdm_loader = tqdm(self.test_dataset, desc='Evaluation', leave=True)
        with torch.no_grad():
            for batch, (X, Y) in enumerate(tqdm_loader):
                X, Y = X.to(device), Y.to(device)
                batch_acc = np.zeros(len(self.clients))
                for client_id, client in enumerate(self.clients):
                    X_spec = client.architecture(X, n_sample, use_spec=True, use_aux=False)
                    batch_acc[client_id] = compute_accuracy(X_spec, Y).item()
                acc += batch_acc * X.shape[0]
                n_test += X.shape[0]
                tqdm_loader.set_postfix_str(f'Batch accuracy = {batch_acc}, Total acc= {acc}')
            acc /= n_test
        return acc

    def eval_aux(self, n_sample=5):
        acc = 0.0
        n_test = 0
        tqdm_loader = tqdm(self.test_dataset, desc='Evaluation', leave=True)
        with torch.no_grad():
            for batch, (X, Y) in enumerate(tqdm_loader):
                X, Y = X.to(device), Y.to(device)
                X_aux = self.server.architecture(X, n_sample, use_spec=False, use_aux=True)
                batch_acc = compute_accuracy(X_aux, Y).item()
                acc += batch_acc * X.shape[0]
                n_test += X.shape[0]
                tqdm_loader.set_postfix_str(f'Batch acc={batch_acc}, Total acc={acc}')
            acc /= n_test
        return acc

    def warm_start(self, n_epoch, n_sample, n_batch):
        print('Warm start')
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}')
            # Local update
            for client_id, client in enumerate(self.clients):
                client.aux_update(self.server.architecture, n_sample, n_batch)
            # Fed Avg
            self.server.fed_average([client.architecture.state_dict() for client in self.clients])
            print_cuda_memory(self.device)
        if self.meta_task:
            print(f'Warm start auxiliary accuracy = {self.eval_aux_meta(n_sample)}')
        else:
            print(f'Warm start auxiliary accuracy = {self.eval_aux(n_sample)}')
        self.save_model(learning_curve=[], prefix='warm-start', date=False)

    def finetune(self, n_epoch=1000, chk_point=10, n_sample=5, n_batch=10):
        best_acc = 0.0
        learning_curve = []
        training_time = []
        print('FedPNAS finetuning')
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}')
            client_time = np.zeros(len(self.clients))
            for client_id, client in enumerate(self.clients):
                start_time.record()
                client.local_finetune(self.server.architecture, n_sample, n_batch)
                end_time.record()
                torch.cuda.synchronize()
                client_time[client_id] = start_time.elapsed_time(end_time)
            training_time.append({'client_time': client_time, 'server_time': 0.0})
            if (epoch + 1) % chk_point == 0:
                if self.meta_task:
                    acc = self.eval_meta(n_sample)
                else:
                    acc = self.eval(n_sample)
                learning_curve.append({'epoch': epoch, 'acc': acc})
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    self.save_model(learning_curve, prefix='fednas-finetune-', time=training_time)
                print(f'Epoch {epoch} accuracy: {acc}')
            print_cuda_memory(self.device)

    def train(self, n_epoch=1000, chk_point=10, n_sample=5, n_batch=10, meta=True):
        best_acc = 0.0
        learning_curve = []
        training_time = []
        print('FedPNAS training')
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}')
            client_time = np.zeros(len(self.clients))
            for client_id, client in enumerate(self.clients):
                start_time.record()
                client.local_update(self.server.architecture, n_sample, n_batch, meta=meta)
                end_time.record()
                torch.cuda.synchronize()
                client_time[client_id] = start_time.elapsed_time(end_time)
            start_time.record()
            self.server.fed_average([client.architecture.state_dict() for client in self.clients])
            end_time.record()
            torch.cuda.synchronize()
            server_time = start_time.elapsed_time(end_time)
            training_time.append({'client_time': client_time, 'server_time': server_time})

            if (epoch + 1) % chk_point == 0:
                if self.meta_task:
                    acc = self.eval_meta(n_sample)
                else:
                    acc = self.eval(n_sample)
                learning_curve.append({'epoch': epoch, 'acc': acc})
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    self.save_model(learning_curve, prefix='fednas-', time=training_time)
                print(f'Epoch {epoch} accuracy: {acc}')
            print_cuda_memory(self.device)

    def save_model(self, learning_curve, prefix='fednas-', date=True, time=None):
        curr_date = str(datetime.now()).replace('.', '-').replace(':', '-').replace(' ', '-') if date else ''
        save_content = {
            'learning_curve': learning_curve,
            'model': self.server.architecture.state_dict(),
            'time': time
        }
        torch.save(save_content, f'{self.save_folder}/{prefix}{curr_date}.pth')

    def load_model(self, model_chkpoint_path):
        model_chkpoint = torch.load(model_chkpoint_path)
        self.server.architecture.load_state_dict(model_chkpoint['model'])
        for client in self.clients:
            client.architecture.load_state_dict(deepcopy(model_chkpoint['model']))


def main(config_name):
    torch.manual_seed(2603)
    np.random.seed(2603)
    config = torch.load(config_name)
    if config['device'] is not None:
        torch.cuda.set_device(config['device'])
    n_client, n_base, n_spec = config['n_client'], config['n_base'], config['n_spec']
    if config['dataset'] == 'cifar':
        Xs, Ys, Xt, Yt, classes = load_cifar(n_client, meta_task=config['meta_task'])
    else:
        Xs, Ys, Xt, Yt, classes = load_mnist(n_client, meta_task=config['meta_task'], mode=config['meta_task_type'])
    orig_channel, n_channel, n_ops, n_class = Xs[0].shape[1], config['n_channel'], len(list(FULL_OPS.keys())), len(classes)
    genotype = CellGenotype(n_hidden=config['n_hidden'], n_sink=config['n_sink'], max_skip=config['max_skip'])

    coordinator = FedPNAS(
        orig_channel, n_channel, n_client, n_base, n_spec, n_ops, n_class, genotype, config['device'], Xs, Ys, Xt, Yt,
        sampler_type=config['sampler_type'], save_folder=config['save_folder'], meta_task=config['meta_task']
    )
    if config['finetune_config'] is None:
        if config['warm_start_load'] is not None:
            coordinator.load_model(config['warm_start_load'])
        else:
            coordinator.warm_start(
                n_epoch=config['warm_start_epoch'],
                n_sample=config['n_sample'],
                n_batch=config['n_batch'])

        coordinator.train(
            n_epoch=config['fed_epoch'],
            chk_point=config['eval_interval'],
            n_sample=config['n_sample'],
            n_batch=config['n_batch'],
            meta=config['meta']
        )
    else:
        finetune_config = config['finetune_config']
        coordinator.load_model(finetune_config['chkpoint'])
        coordinator.finetune(
            n_epoch=finetune_config['n_epoch'],
            chk_point=config['eval_interval'],
            n_sample=config['n_sample'],
            n_batch=config['n_batch'],
        )


def create_config(config_name):
    torch.save({
        'dataset': 'mnist',
        'n_client': 5,
        'n_base': 3,
        'n_spec': 1,
        'n_channel': 16,
        'n_hidden': 3,
        'n_sink': 1,
        'max_skip': 3,
        'sampler_type': 'cell_based',
        'save_folder': '',
        'warm_start_load': None,
        'warm_start_epoch': 100,
        'fed_epoch': 1000,
        'n_batch': 10,
        'n_sample': 5,
        'eval_interval': 5,
        'device': 1,
        'meta': True,
        'meta_task': True,
        'meta_task_type': 'mixed',
        'finetune_config': None
    }, config_name)


if __name__ == '__main__':
    config_name = ''
    create_config(config_name)
    main(config_name)

