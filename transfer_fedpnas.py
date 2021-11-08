from architecture import *
from server import *
from client import *
from cifar_utils import *
from mnist_utils import *


# Create a new client, adapt with limited data
class TransferFedPNAS:
    def __init__(self, orig_channel, n_channel, n_base, n_spec, n_ops, n_class, genotype, gpu_device, rotation,
                 Xs, Ys, Xt, Yt, sampler_type='cell_based', save_folder='./model_chkpoint'):
        architecture = SNAS_Architecture(
            orig_channel, n_channel, genotype, n_base, n_spec,
            n_ops, n_class, sampler_type=sampler_type
        )
        self.rotation = rotation
        self.save_folder = save_folder
        self.device = gpu_device
        self.server = Server(architecture)
        self.transfer_client = Client(architecture, client_id=0)
        self.transfer_client.load_data(Xs, Ys)
        self.test_dataset = DataLoader(TensorDataset(Xt, Yt), batch_size=BATCH_SIZE, shuffle=True)

    def eval(self, n_sample=5):
        acc = 0.0
        n_test = 0
        tqdm_loader = tqdm(self.test_dataset, desc='Evaluation', leave=True)
        with torch.no_grad():
            for batch, (X, Y) in enumerate(tqdm_loader):
                X, Y = X.to(device), Y.to(device)
                X_spec = self.transfer_client.architecture(X, n_sample, use_spec=True, use_aux=False)
                batch_acc = compute_accuracy(X_spec, Y).item()
                acc += batch_acc * X.shape[0]
                n_test += X.shape[0]
                tqdm_loader.set_postfix_str(f'Batch accuracy = {batch_acc}, Total acc= {acc}')
            acc /= n_test
        return acc

    def finetune(self, n_epoch=1000, chk_point=10, n_sample=5, n_batch=10):
        best_acc = 0.0
        learning_curve = []
        training_time = []
        print('FedPNAS transfer finetuning')
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}')
            start_time.record()
            self.transfer_client.local_finetune(self.server.architecture, n_sample, n_batch)
            end_time.record()
            torch.cuda.synchronize()
            client_time = start_time.elapsed_time(end_time)
            training_time.append({'client_time': client_time, 'server_time': 0.0})
            if (epoch + 1) % chk_point == 0:
                acc = self.eval()
                learning_curve.append({'epoch': epoch, 'acc': acc})
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    self.save_model(learning_curve, prefix=f'fednas-transfer-{self.rotation}-', time=training_time)
                print(f'Epoch {epoch} accuracy: {acc}')
            print_cuda_memory(self.device)

    def load_model(self, model_chkpoint_path):
        model_chkpoint = torch.load(model_chkpoint_path)
        self.transfer_client.architecture.load_state_dict(model_chkpoint['model'])

    def save_model(self, learning_curve, prefix='fednas-', date=True, time=None):
        curr_date = str(datetime.now()).replace('.', '-').replace(':', '-').replace(' ', '-') if date else ''
        save_content = {
            'learning_curve': learning_curve,
            'model': self.transfer_client.architecture.state_dict(),
            'time': time
        }
        torch.save(save_content, f'{self.save_folder}/{prefix}{curr_date}.pth')


def main(config_name):
    torch.manual_seed(2603)
    np.random.seed(2603)
    config = torch.load(config_name)
    if config['device'] is not None:
        torch.cuda.set_device(config['device'])
    n_base, n_spec = config['n_base'], config['n_spec']
    if config['dataset'] == 'cifar':
        Xs, Ys, Xt, Yt, classes = load_cifar_transfer(deg=config['rotation'])
    else:
        Xs, Ys, Xt, Yt, classes = load_mnist_transfer(deg=config['rotation'])
    orig_channel, n_channel, n_ops, n_class = Xs.shape[1], config['n_channel'], len(list(FULL_OPS.keys())), len(classes)
    genotype = CellGenotype(n_hidden=config['n_hidden'], n_sink=config['n_sink'], max_skip=config['max_skip'])

    coordinator = TransferFedPNAS(
        orig_channel, n_channel, n_base, n_spec, n_ops, n_class, genotype, config['device'], config['rotation'],
        Xs, Ys, Xt, Yt, sampler_type=config['sampler_type'], save_folder=config['save_folder']
    )

    coordinator.load_model(config['chkpoint'])
    coordinator.finetune(
        n_epoch=config['n_epoch'],
        chk_point=config['eval_interval'],
        n_sample=config['n_sample'],
        n_batch=config['n_batch'],
    )


def create_config(config_name):
    torch.save({
        'dataset': 'mnist',
        'n_base': 3,
        'n_spec': 1,
        'n_channel': 16,
        'n_hidden': 3,
        'n_sink': 1,
        'max_skip': 3,
        'sampler_type': 'cell_based',
        'save_folder': '',
        'chkpoint': '',
        'rotation': 90,
        'n_batch': 10,
        'n_epoch': 100,
        'n_sample': 5,
        'eval_interval': 5,
        'device': 1
    }, config_name)


if __name__ == '__main__':
    config_name = './model_chkpoint_mnist_transfer_std/config_mnist_transfer_std.pth'
    create_config(config_name)
    main(config_name)

